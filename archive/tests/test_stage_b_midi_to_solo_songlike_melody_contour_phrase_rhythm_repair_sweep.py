from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    DELIVERY_SCHEMA_VERSION,
    LISTENING_GAP_SCHEMA_VERSION,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as FINAL_STATUS_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    build_quality_rubric_baseline_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_repair_followup import (
    BOUNDARY as FOLLOWUP_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as FOLLOWUP_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SELECTED_TARGET as FOLLOWUP_SELECTED_TARGET,
    SCHEMA_VERSION as FOLLOWUP_SCHEMA_VERSION,
    TIE_TARGET_LABELS,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
    SCHEMA_VERSION as POST_MVP_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError,
    build_songlike_melody_contour_phrase_rhythm_repair_sweep_report,
    validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep import (
    BOUNDARY as SOURCE_SWEEP_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SWEEP_SCHEMA_VERSION,
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
    "followup_objective_source_outside_soloing_schema_context_preserved": True,
    "followup_objective_source_outside_soloing_objective_schema_version": (
        OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
    ),
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_schema_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_objective_schema_version": (
        OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
    ),
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_schema_context_preserved": True,
    "repair_sweep_source_outside_soloing_objective_schema_version": (
        OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
    ),
}


def post_mvp_quality_plan() -> dict:
    return {
        "schema_version": POST_MVP_SCHEMA_VERSION,
        "boundary": POST_MVP_BOUNDARY,
        "source_schema_versions": {
            "final_status_audit": FINAL_STATUS_SCHEMA_VERSION,
            "delivery_package": DELIVERY_SCHEMA_VERSION,
            "listening_review_quality_gap": LISTENING_GAP_SCHEMA_VERSION,
            "quality_gap_decision": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "current_evidence": CURRENT_EVIDENCE_SCHEMA_VERSION,
        },
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
            "source_final_status_schema_version": FINAL_STATUS_SCHEMA_VERSION,
            "source_delivery_package_schema_version": DELIVERY_SCHEMA_VERSION,
            "source_listening_gap_schema_version": LISTENING_GAP_SCHEMA_VERSION,
            "source_quality_gap_schema_version": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "source_current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "outside_soloing_repair_evidence_ready": True,
            "outside_soloing_repair_source_context_preserved": True,
            "outside_soloing_repair_schema_context_preserved": True,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
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
            **SOURCE_CONTEXT,
        },
        "readiness": {
            "post_mvp_quality_iteration_plan_completed": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
            "outside_soloing_repair_source_context_preserved": True,
            "outside_soloing_repair_schema_context_preserved": True,
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
    objective_context = {
        "objective_source_outside_soloing_repair_evidence_ready": True,
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
        "objective_source_outside_soloing_not_evaluable_count": 6,
        "objective_repaired_outside_soloing_not_evaluable_count": 6,
        **{f"objective_{key}": value for key, value in SOURCE_CONTEXT.items()},
    }
    repair_sweep_context = {
        "repair_sweep_source_outside_soloing_repair_evidence_ready": True,
        "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
        "repair_sweep_source_outside_soloing_repair_source_context_preserved": True,
        "repair_sweep_source_outside_soloing_repair_schema_context_preserved": True,
        "repair_sweep_source_outside_soloing_repair_objective_schema_version": (
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
        ),
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
        "repair_sweep_source_outside_soloing_repair_source_targeted": False,
        "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved": True,
        "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
        "repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta": 2,
        "repair_sweep_source_outside_soloing_not_evaluable_count": 6,
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": 6,
        **{f"repair_sweep_{key}": value for key, value in SOURCE_CONTEXT.items()},
    }
    return {
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
        "boundary": FOLLOWUP_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_songlike_melody_contour_repair_objective_only_next_decision",
        "repair_sweep_boundary": SOURCE_SWEEP_BOUNDARY,
        "source_schema_versions": dict(FOLLOWUP_SOURCE_SCHEMA_VERSIONS),
        "repair_sweep_summary": {
            "remaining_failure_counts": {
                "phrase_shape_missing_tension_release": 2,
                "rhythmic_monotony": 2,
            },
            "not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
        },
        "selected_next_target": {
            "selected_target": FOLLOWUP_SELECTED_TARGET,
            "selected_next_boundary": FOLLOWUP_NEXT_BOUNDARY,
            "primary_remaining_failure_labels": list(TIE_TARGET_LABELS),
            "primary_remaining_failure_count": 2,
        },
        "followup_targets": {
            "primary_labels": list(TIE_TARGET_LABELS),
            "secondary_failure_counts": {
                "phrase_shape_missing_tension_release": 2,
                "rhythmic_monotony": 2,
            },
            "not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
            **objective_context,
            **repair_sweep_context,
        },
        "readiness": {
            "followup_decision_completed": True,
            "phrase_rhythm_tie_target_selected": True,
            "candidate_count": 6,
            "source_total_failure_label_count": 8,
            "repaired_total_failure_label_count": 4,
            "failure_label_delta": 4,
            "technical_regression_count": 0,
            **objective_context,
            **repair_sweep_context,
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


def source_repair_sweep(*, technical_regression_count: int = 0) -> dict:
    labels = [
        [],
        ["phrase_shape_missing_tension_release", "rhythmic_monotony"],
        [],
        ["rhythmic_monotony"],
        [],
        ["phrase_shape_missing_tension_release"],
    ]
    return {
        "schema_version": SOURCE_SWEEP_SCHEMA_VERSION,
        "boundary": SOURCE_SWEEP_BOUNDARY,
        "candidate_repairs": [
            {
                "source": "test",
                "rank": index + 1,
                "contour_repaired_midi_path": f"source_{index + 1}.mid",
                "contour_repaired_labeling": {
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
            "source_total_failure_label_count": 8,
            "repaired_total_failure_label_count": 4,
            "failure_label_delta": 4,
            "source_songlike_failure_count": 5,
            "repaired_songlike_failure_count": 0,
            "songlike_failure_delta": 5,
            "improved_candidate_count": 4,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": {
                "phrase_shape_missing_tension_release": 2,
                "rhythmic_monotony": 2,
            },
            "source_outside_soloing_repair_evidence_ready": True,
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
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            **SOURCE_CONTEXT,
            "target_supported": True,
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
    }


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepTest(
    unittest.TestCase
):
    def test_reduces_phrase_rhythm_labels_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                followup_decision=followup_decision(),
                source_repair_sweep=source_repair_sweep(),
                rubric_baseline=rubric_baseline(root),
                output_dir=root / "phrase_rhythm_repair",
                issue_number=1112,
            )
            summary = validate_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_candidate_count=6,
                require_sweep_completed=True,
                require_target_supported=True,
                require_phrase_rhythm_delta=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(
                summary[
                    "songlike_melody_contour_phrase_rhythm_repair_sweep_completed"
                ]
            )
            self.assertTrue(
                summary[
                    "songlike_melody_contour_phrase_rhythm_repair_target_supported"
                ]
            )
            self.assertEqual(summary["candidate_count"], 6)
            self.assertEqual(summary["source_phrase_rhythm_failure_count"], 4)
            self.assertLess(summary["repaired_phrase_rhythm_failure_count"], 4)
            self.assertGreater(summary["phrase_rhythm_failure_delta"], 0)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                summary["source_songlike_melody_contour_repair_followup_schema_version"],
                FOLLOWUP_SCHEMA_VERSION,
            )
            self.assertEqual(
                summary["source_songlike_melody_contour_repair_sweep_schema_version"],
                SOURCE_SWEEP_SCHEMA_VERSION,
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
            self.assertTrue(summary["source_outside_soloing_repair_source_context_preserved"])
            self.assertTrue(summary["source_outside_soloing_repair_schema_context_preserved"])
            self.assertEqual(
                summary["source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[f"objective_{key}"], SOURCE_CONTEXT[key])
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertEqual(
                summary[
                    "source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "source_outside_soloing_repair_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_delta"],
                3,
            )
            self.assertFalse(summary["source_outside_soloing_repair_source_targeted"])
            self.assertTrue(
                summary["source_outside_soloing_repair_source_residual_risk_preserved"]
            )
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repaired_not_evaluable_counts"]["outside_soloing_without_context"],
                6,
            )
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_followup_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(quality_claim=True),
                    source_repair_sweep=source_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_source_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source_repair_sweep(technical_regression_count=1),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_followup_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = followup_decision()
            source["source_schema_versions"][
                "songlike_melody_contour_repair_objective_next"
            ] = "stale"
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=source,
                    source_repair_sweep=source_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_source_sweep_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = source_repair_sweep()
            source["schema_version"] = "stale"
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = source_repair_sweep()
            source["aggregate"]["source_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_source_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = source_repair_sweep()
            source["aggregate"][
                "source_outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 1
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_missing_bridge_source_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = source_repair_sweep()
            source["aggregate"].pop(BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS[0])
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_bridge_source_context_preservation_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = source_repair_sweep()
            source["aggregate"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=followup_decision(),
                    source_repair_sweep=source,
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_source_context_preservation_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = followup_decision()
            source["readiness"][
                "repair_sweep_source_outside_soloing_repair_source_context_preserved"
            ] = False
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=source,
                    source_repair_sweep=source_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_rejects_schema_context_preservation_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = followup_decision()
            source["readiness"][
                "repair_sweep_source_outside_soloing_repair_schema_context_preserved"
            ] = False
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairSweepError
            ):
                build_songlike_melody_contour_phrase_rhythm_repair_sweep_report(
                    followup_decision=source,
                    source_repair_sweep=source_repair_sweep(),
                    rubric_baseline=rubric_baseline(Path(tmp)),
                    output_dir=Path(tmp) / "phrase_rhythm_repair",
                    issue_number=1112,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_repair_audio_package",
        )
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_v5",
        )


if __name__ == "__main__":
    unittest.main()
