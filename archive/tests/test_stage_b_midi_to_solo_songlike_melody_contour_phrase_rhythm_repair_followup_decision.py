from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup import (
    BOUNDARY,
    CONTEXT_TARGET_LABELS,
    EXPECTED_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    SCHEMA_VERSION,
    StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SCHEMA_VERSION as OBJECTIVE_NEXT_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as SWEEP_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as SWEEP_NEXT_BOUNDARY,
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def objective_next_report(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": OBJECTIVE_NEXT_SCHEMA_VERSION,
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "source_schema_versions": dict(OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS),
        "objective_summary": {
            "source_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard"
                ]
            ),
            "source_songlike_melody_contour_phrase_rhythm_repair_listening_review_package_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_phrase_rhythm_repair_listening_review_package"
                ]
            ),
            "source_songlike_melody_contour_phrase_rhythm_repair_audio_package_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_phrase_rhythm_repair_audio_package"
                ]
            ),
            "source_songlike_melody_contour_phrase_rhythm_repair_sweep_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_phrase_rhythm_repair_sweep"
                ]
            ),
            "source_songlike_melody_contour_repair_followup_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_followup_decision"
                ]
            ),
            "source_songlike_melody_contour_repair_objective_next_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_objective_next"
                ]
            ),
            "source_songlike_melody_contour_repair_sweep_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_sweep"
                ]
            ),
            "source_songlike_melody_contour_repair_listening_review_input_guard_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_listening_review_input_guard"
                ]
            ),
            "source_songlike_melody_contour_repair_listening_review_package_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_listening_review_package"
                ]
            ),
            "source_songlike_melody_contour_repair_audio_package_schema_version": (
                OBJECTIVE_NEXT_SOURCE_SCHEMA_VERSIONS[
                    "songlike_melody_contour_repair_audio_package"
                ]
            ),
            "review_item_count": 6,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "failure_label_delta": 3,
            "phrase_rhythm_failure_delta": 3,
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_context_preserved": True,
            "objective_source_outside_soloing_repair_schema_context_preserved": True,
            "objective_source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            **{f"objective_{key}": value for key, value in SOURCE_CONTEXT.items()},
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "source_outside_soloing_repair_source_context_preserved": True,
            "source_outside_soloing_repair_schema_context_preserved": True,
            "source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_source_targeted": False,
            "source_outside_soloing_repair_source_residual_risk_preserved": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_repair_pitch_role_risk_delta": 2,
            **SOURCE_CONTEXT,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "repaired_not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
            "current_quality_claim_ready": False,
        },
        "readiness": {
            "objective_next_decision_completed": True,
            "phrase_rhythm_followup_required": True,
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
        "schema_version": SWEEP_SCHEMA_VERSION,
        "boundary": SWEEP_BOUNDARY,
        "source_schema_versions": dict(SWEEP_SOURCE_SCHEMA_VERSIONS),
        "candidate_repairs": [
            {
                "phrase_rhythm_repaired_labeling": {
                    "failure_labels": [],
                    "not_evaluable_labels": list(CONTEXT_TARGET_LABELS),
                }
            }
            for _ in range(6)
        ],
        "aggregate": {
            "candidate_count": 6,
            "target_supported": True,
            "source_total_failure_label_count": 4,
            "repaired_total_failure_label_count": 1,
            "failure_label_delta": 3,
            "source_phrase_rhythm_failure_count": 4,
            "repaired_phrase_rhythm_failure_count": 1,
            "phrase_rhythm_failure_delta": 3,
            "improved_candidate_count": 2,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": {
                "rhythmic_monotony": 1,
            },
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_context_preserved": True,
            "objective_source_outside_soloing_repair_schema_context_preserved": True,
            "objective_source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            **{f"objective_{key}": value for key, value in SOURCE_CONTEXT.items()},
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "source_outside_soloing_repair_source_context_preserved": True,
            "source_outside_soloing_repair_schema_context_preserved": True,
            "source_outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_source_targeted": False,
            "source_outside_soloing_repair_source_residual_risk_preserved": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_repair_pitch_role_risk_delta": 2,
            **SOURCE_CONTEXT,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "repaired_not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
        },
        "readiness": {
            "songlike_melody_contour_phrase_rhythm_repair_sweep_completed": True,
            "songlike_melody_contour_phrase_rhythm_repair_target_supported": True,
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
            "next_boundary": SWEEP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionTest(unittest.TestCase):
    def test_selects_chord_context_pitch_role_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=1206,
            )
            summary = validate_followup_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_followup_decision=True,
                require_context_pitch_role_bridge=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["followup_decision_completed"])
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1206)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                summary[
                    "source_songlike_melody_contour_phrase_rhythm_repair_objective_next_schema_version"
                ],
                OBJECTIVE_NEXT_SCHEMA_VERSION,
            )
            self.assertEqual(
                summary[
                    "source_songlike_melody_contour_phrase_rhythm_repair_sweep_schema_version"
                ],
                SWEEP_SCHEMA_VERSION,
            )
            for key, expected in EXPECTED_SOURCE_SCHEMA_VERSIONS.items():
                self.assertEqual(report["source_schema_versions"][key], expected)
            self.assertTrue(summary["chord_context_pitch_role_bridge_selected"])
            self.assertEqual(
                summary["context_target_labels"],
                list(CONTEXT_TARGET_LABELS),
            )
            self.assertEqual(summary["primary_remaining_failure_labels"], ["rhythmic_monotony"])
            self.assertEqual(summary["primary_remaining_failure_count"], 1)
            self.assertEqual(summary["failure_label_delta"], 3)
            self.assertEqual(summary["phrase_rhythm_failure_delta"], 3)
            self.assertEqual(summary["context_not_evaluable_min_count"], 6)
            self.assertTrue(summary["objective_source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_pitch_role_risk_count_after"],
                0,
            )
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertTrue(
                summary["objective_source_outside_soloing_repair_source_context_preserved"]
            )
            self.assertTrue(
                summary["objective_source_outside_soloing_repair_schema_context_preserved"]
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[f"objective_{key}"], SOURCE_CONTEXT[key])
            for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
                self.assertTrue(summary[f"objective_{key}"])
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
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(summary["objective_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["objective_repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertTrue(summary["repair_sweep_source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after"],
                0,
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_repair_source_context_preserved"]
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_repair_schema_context_preserved"]
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[f"repair_sweep_{key}"], SOURCE_CONTEXT[key])
            for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS:
                self.assertTrue(summary[f"repair_sweep_{key}"])
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta"],
                3,
            )
            self.assertFalse(summary["repair_sweep_source_outside_soloing_repair_source_targeted"])
            self.assertTrue(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved"
                ]
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(summary["repair_sweep_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repair_sweep_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=repair_sweep_report(technical_regression_count=1),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_missing_objective_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = objective_next_report()
            del source["objective_summary"]["source_outside_soloing_not_evaluable_count"]

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=source,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_missing_repair_sweep_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            del source["aggregate"]["repaired_outside_soloing_not_evaluable_count"]

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_repair_sweep_source_context_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"]["source_outside_soloing_repair_source_pitch_role_risk_delta"] = 1

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_missing_bridge_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = objective_next_report()
            source["objective_summary"].pop(BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS[0])

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=source,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_source_context_preservation_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = objective_next_report()
            source["objective_summary"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=source,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_repair_sweep_source_context_preservation_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_objective_repair_sweep_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"]["repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"] = 4

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_objective_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = objective_next_report()
            source["source_schema_versions"][
                "songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard"
            ] = "stale"

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=source,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_rejects_repair_sweep_schema_context_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"][
                "source_outside_soloing_repair_schema_context_preserved"
            ] = False

            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1206,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge")
        self.assertEqual(SELECTED_TARGET, "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge")
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision_v5",
        )


if __name__ == "__main__":
    unittest.main()
