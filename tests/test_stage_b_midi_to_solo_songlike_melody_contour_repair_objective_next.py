from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_repair_objective_next import (
    BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_input import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY as SOURCE_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SCHEMA_VERSION as SOURCE_GUARD_SCHEMA_VERSION,
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
        "schema_version": SOURCE_GUARD_SCHEMA_VERSION,
        "boundary": SOURCE_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package",
        "source_schema_versions": dict(EXPECTED_SOURCE_SCHEMA_VERSIONS),
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "review_item_count": 6,
            "required_input_field_count": 4,
            "source_summary": {
                "technical_wav_validation": True,
                "rendered_audio_file_count": 6,
                "sample_rate": 44100,
                "duration_min_seconds": 18.849,
                "duration_max_seconds": 18.992,
                "failure_label_delta": 4,
                "source_songlike_failure_count": 5,
                "repaired_songlike_failure_count": 0,
                "songlike_failure_delta": 5,
                "source_outside_soloing_repair_evidence_ready": True,
                "objective_source_outside_soloing_repair_wav_count": 6,
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


class StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_pending_review_to_followup_decision_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_objective_next_report(
                input_guard_report=input_guard_report(),
                output_dir=root / "objective_next",
                issue_number=1192,
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
            self.assertEqual(report["issue_number"], 1192)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                report["source_schema_versions"][
                    "songlike_melody_contour_repair_listening_review_input_guard"
                ],
                SOURCE_GUARD_SCHEMA_VERSION,
            )
            self.assertEqual(
                summary[
                    "source_songlike_melody_contour_repair_listening_review_input_guard_schema_version"
                ],
                SOURCE_GUARD_SCHEMA_VERSION,
            )
            for key, expected in EXPECTED_SOURCE_SCHEMA_VERSIONS.items():
                self.assertEqual(report["source_schema_versions"][key], expected)
            self.assertTrue(summary["objective_next_decision_completed"])
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertEqual(summary["failure_label_delta"], 4)
            self.assertEqual(summary["songlike_failure_delta"], 5)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(summary["objective_source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
                ],
                5,
            )
            self.assertTrue(summary["objective_source_outside_soloing_repair_source_context_preserved"])
            self.assertTrue(
                summary["objective_source_outside_soloing_repair_schema_context_preserved"]
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[f"objective_{key}"], SOURCE_CONTEXT[key])
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
                summary["objective_source_outside_soloing_repair_source_residual_risk_preserved"]
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_pitch_role_risk_count_after"],
                0,
            )
            self.assertEqual(summary["objective_source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(
                summary["source_outside_soloing_repair_source_objective_pitch_role_risk_count"],
                5,
            )
            self.assertTrue(summary["source_outside_soloing_repair_source_context_preserved"])
            self.assertTrue(summary["source_outside_soloing_repair_schema_context_preserved"])
            self.assertEqual(
                summary["source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_before"],
                5,
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_after"],
                2,
            )
            self.assertEqual(summary["source_outside_soloing_repair_source_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_source_targeted"])
            self.assertTrue(summary["source_outside_soloing_repair_source_residual_risk_preserved"])
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertTrue(summary["songlike_contour_followup_required"])
            self.assertFalse(summary["current_quality_claim_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=input_guard_report(quality_claim=True),
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"]["repaired_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_source_context_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "source_outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 9
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_missing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_false_source_context_preserved_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["source_schema_versions"][
                "songlike_melody_contour_repair_listening_review_package"
            ] = "stale"
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_rejects_false_outside_soloing_schema_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "source_outside_soloing_repair_schema_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1192,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_objective_only_next_decision")
        self.assertEqual(FOLLOWUP_DECISION_NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision")
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_repair_objective_next_v5",
        )


if __name__ == "__main__":
    unittest.main()
