from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    build_quality_rubric_baseline_report,
)
from scripts.decide_stage_b_midi_to_solo_targeted_quality_repair_followup import (
    BOUNDARY as FOLLOWUP_BOUNDARY,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SELECTED_TARGET as FOLLOWUP_SELECTED_TARGET,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    SONGLIKE_LABEL,
    StageBMidiToSoloSonglikeMelodyContourRepairSweepError,
    build_songlike_melody_contour_repair_sweep_report,
    validate_songlike_melody_contour_repair_sweep_report,
)
from scripts.run_stage_b_midi_to_solo_targeted_quality_repair_sweep import (
    BOUNDARY as TARGETED_REPAIR_SWEEP_BOUNDARY,
)


def post_mvp_quality_plan() -> dict:
    return {
        "boundary": POST_MVP_BOUNDARY,
        "selected_next_target": {
            "selected_target": POST_MVP_SELECTED_TARGET,
            "selected_next_boundary": POST_MVP_NEXT_BOUNDARY,
        },
        "ordered_work": [
            {"target": "quality_rubric_baseline"},
            {"target": "candidate_failure_labeling"},
            {"target": "targeted_quality_repair_sweep"},
            {"target": "audio_review_package"},
        ],
        "quality_failure_taxonomy_seed": [f"failure_{index}" for index in range(7)],
        "post_mvp_status": {
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "outside_soloing_repair_evidence_ready": True,
            "outside_soloing_repair_wav_count": 6,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "outside_soloing_repair_source_targeted": False,
            "outside_soloing_repair_source_residual_risk_preserved": True,
            "outside_soloing_repair_pitch_role_risk_count_after": 0,
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "post_mvp_quality_iteration_plan_completed": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
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
            "next_boundary": POST_MVP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def rubric_baseline(root: Path) -> dict:
    return build_quality_rubric_baseline_report(
        post_mvp_quality_plan=post_mvp_quality_plan(),
        output_dir=root / "rubric",
        issue_number=746,
    )


def followup_decision(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": FOLLOWUP_BOUNDARY,
        "repair_sweep_summary": {
            "remaining_failure_counts": {
                "dead_air_or_density_gap": 1,
                "phrase_shape_missing_tension_release": 2,
                SONGLIKE_LABEL: 5,
            }
        },
        "selected_next_target": {
            "selected_target": FOLLOWUP_SELECTED_TARGET,
            "selected_next_boundary": FOLLOWUP_NEXT_BOUNDARY,
            "dominant_remaining_failure_label": SONGLIKE_LABEL,
            "dominant_remaining_failure_count": 5,
        },
        "readiness": {
            "followup_decision_completed": True,
            "dominant_songlike_target_selected": True,
            "candidate_count": 6,
            "source_total_failure_label_count": 12,
            "repaired_total_failure_label_count": 8,
            "failure_label_delta": 4,
            "technical_regression_count": 0,
            "objective_source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_wav_count": 6,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "objective_source_outside_soloing_not_evaluable_count": 6,
            "objective_repaired_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_source_outside_soloing_repair_evidence_ready": True,
            "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "repair_sweep_source_outside_soloing_repair_source_targeted": False,
            "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "repair_sweep_source_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": 6,
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
            "next_boundary": FOLLOWUP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def targeted_repair_sweep(*, technical_regression_count: int = 0) -> dict:
    labels = [
        [SONGLIKE_LABEL],
        [SONGLIKE_LABEL],
        [SONGLIKE_LABEL],
        ["phrase_shape_missing_tension_release", SONGLIKE_LABEL],
        [],
        ["dead_air_or_density_gap", "phrase_shape_missing_tension_release", SONGLIKE_LABEL],
    ]
    return {
        "boundary": TARGETED_REPAIR_SWEEP_BOUNDARY,
        "candidate_repairs": [
            {
                "source": "test",
                "rank": index + 1,
                "repaired_midi_path": f"source_{index + 1}.mid",
                "repaired_labeling": {
                    "failure_labels": row_labels,
                    "not_evaluable_labels": [
                        "outside_soloing_without_context",
                        "weak_chord_tone_landing",
                    ],
                },
            }
            for index, row_labels in enumerate(labels)
        ],
        "aggregate": {
            "candidate_count": 6,
            "source_total_failure_label_count": 12,
            "repaired_total_failure_label_count": 8,
            "failure_label_delta": 4,
            "improved_candidate_count": 4,
            "technical_regression_count": technical_regression_count,
            "source_outside_soloing_repair_evidence_ready": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "repaired_failure_counts": {
                "dead_air_or_density_gap": 1,
                "phrase_shape_missing_tension_release": 2,
                SONGLIKE_LABEL: 5,
            },
        },
        "readiness": {
            "targeted_quality_repair_sweep_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
    }


class StageBMidiToSoloSonglikeMelodyContourRepairSweepTest(unittest.TestCase):
    def test_reduces_songlike_label_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_songlike_melody_contour_repair_sweep_report(
                followup_decision=followup_decision(),
                targeted_repair_sweep=targeted_repair_sweep(),
                rubric_baseline=rubric_baseline(root),
                output_dir=root / "songlike_repair",
                issue_number=762,
            )
            summary = validate_songlike_melody_contour_repair_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_candidate_count=6,
                require_sweep_completed=True,
                require_target_supported=True,
                require_songlike_delta=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["songlike_melody_contour_repair_sweep_completed"])
            self.assertTrue(summary["songlike_melody_contour_repair_target_supported"])
            self.assertEqual(summary["candidate_count"], 6)
            self.assertEqual(summary["source_songlike_failure_count"], 5)
            self.assertLess(summary["repaired_songlike_failure_count"], 5)
            self.assertGreater(summary["songlike_failure_delta"], 0)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(summary["objective_source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
                ],
                5,
            )
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
            self.assertEqual(
                summary["repaired_not_evaluable_counts"]["outside_soloing_without_context"],
                6,
            )
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertEqual(
                summary["next_recommended_issue"],
                "Stage B MIDI-to-solo songlike melody contour repair audio package source-context refresh",
            )
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_followup_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairSweepError):
                build_songlike_melody_contour_repair_sweep_report(
                    followup_decision=followup_decision(quality_claim=True),
                    targeted_repair_sweep=targeted_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "songlike_repair",
                    issue_number=762,
                )

    def test_rejects_source_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairSweepError):
                build_songlike_melody_contour_repair_sweep_report(
                    followup_decision=followup_decision(),
                    targeted_repair_sweep=targeted_repair_sweep(technical_regression_count=1),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "songlike_repair",
                    issue_number=762,
                )

    def test_rejects_missing_followup_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = followup_decision()
            source["readiness"]["repair_sweep_repaired_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairSweepError):
                build_songlike_melody_contour_repair_sweep_report(
                    followup_decision=source,
                    targeted_repair_sweep=targeted_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "songlike_repair",
                    issue_number=846,
                )

    def test_rejects_missing_targeted_sweep_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = targeted_repair_sweep()
            source["aggregate"]["source_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairSweepError):
                build_songlike_melody_contour_repair_sweep_report(
                    followup_decision=followup_decision(),
                    targeted_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "songlike_repair",
                    issue_number=846,
                )

    def test_rejects_followup_source_risk_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = followup_decision()
            source["readiness"][
                "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 1
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourRepairSweepError):
                build_songlike_melody_contour_repair_sweep_report(
                    followup_decision=source,
                    targeted_repair_sweep=targeted_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "songlike_repair",
                    issue_number=932,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_sweep")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_audio_package")
        self.assertEqual(SELECTED_TARGET, "songlike_melody_contour_repair_audio_package")


if __name__ == "__main__":
    unittest.main()
