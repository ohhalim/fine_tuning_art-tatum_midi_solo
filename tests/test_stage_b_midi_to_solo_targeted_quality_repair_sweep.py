from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    DELIVERY_SCHEMA_VERSION,
    LISTENING_GAP_SCHEMA_VERSION,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as FINAL_STATUS_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import BOUNDARY as DELIVERY_BOUNDARY
from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    build_quality_rubric_baseline_report,
    SCHEMA_VERSION as RUBRIC_SCHEMA_VERSION,
)
from scripts.label_stage_b_midi_to_solo_candidate_failures import (
    build_candidate_failure_labeling_report,
    SCHEMA_VERSION as LABELING_SCHEMA_VERSION,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
    SCHEMA_VERSION as POST_MVP_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_targeted_quality_repair_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloTargetedQualityRepairSweepError,
    build_targeted_quality_repair_sweep_report,
    validate_targeted_quality_repair_sweep_report,
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


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.5
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.25)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


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
        issue_number=1168,
    )


def delivery_package(paths: list[Path]) -> dict:
    return {
        "boundary": DELIVERY_BOUNDARY,
        "delivery_package": {
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
        },
        "artifact_manifest": {
            "cli_repaired_midi_candidates": [
                {
                    "rank": index + 1,
                    "repaired_midi_path": str(path),
                    "note_count": 24,
                    "unique_pitch_count": 5,
                    "dead_air_ratio": 0.0,
                    "objective_supported": True,
                }
                for index, path in enumerate(paths[:3])
            ],
            "changed_ratio_repair_audio_candidates": [
                {
                    "rank": index + 1,
                    "repaired_midi_path": str(path),
                    "repaired_unique_pitch_count": 5,
                    "pitch_changed_ratio": 0.3,
                    "repaired_max_interval": 4,
                }
                for index, path in enumerate(paths[3:], start=1)
            ],
        },
        "readiness": {
            "mvp_delivery_package_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
    }


def candidate_failure_labeling(root: Path, *, quality_claim: bool = False) -> dict:
    paths = [root / f"candidate_{index}.mid" for index in range(6)]
    for path in paths:
        write_midi(path, [60, 62, 64, 65] * 6)
    report = build_candidate_failure_labeling_report(
        rubric_baseline=rubric_baseline(root),
        mvp_delivery_package=delivery_package(paths),
        output_dir=root / "labels",
        issue_number=1170,
    )
    if quality_claim:
        report["readiness"]["midi_to_solo_musical_quality_claimed"] = True
    return report


class StageBMidiToSoloTargetedQualityRepairSweepTest(unittest.TestCase):
    def test_repairs_current_failure_labels_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_targeted_quality_repair_sweep_report(
                candidate_failure_labeling=candidate_failure_labeling(root),
                rubric_baseline=rubric_baseline(root),
                output_dir=root / "sweep",
                issue_number=1172,
            )
            summary = validate_targeted_quality_repair_sweep_report(
                report,
                expected_boundary=BOUNDARY,
                min_candidate_count=6,
                require_sweep_completed=True,
                require_failure_delta=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1172)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                summary["source_candidate_failure_labeling_schema_version"],
                LABELING_SCHEMA_VERSION,
            )
            self.assertEqual(summary["source_quality_rubric_schema_version"], RUBRIC_SCHEMA_VERSION)
            self.assertEqual(summary["source_post_mvp_plan_schema_version"], POST_MVP_SCHEMA_VERSION)
            self.assertEqual(summary["source_final_status_schema_version"], FINAL_STATUS_SCHEMA_VERSION)
            self.assertEqual(summary["source_delivery_package_schema_version"], DELIVERY_SCHEMA_VERSION)
            self.assertEqual(summary["source_listening_gap_schema_version"], LISTENING_GAP_SCHEMA_VERSION)
            self.assertEqual(summary["source_quality_gap_schema_version"], QUALITY_GAP_DECISION_SCHEMA_VERSION)
            self.assertEqual(summary["source_current_evidence_schema_version"], CURRENT_EVIDENCE_SCHEMA_VERSION)
            self.assertTrue(summary["targeted_quality_repair_sweep_completed"])
            self.assertTrue(summary["targeted_quality_repair_target_supported"])
            self.assertEqual(summary["candidate_count"], 6)
            self.assertGreater(summary["failure_label_delta"], 0)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertTrue(summary["source_outside_soloing_repair_source_context_preserved"])
            self.assertTrue(summary["source_outside_soloing_repair_schema_context_preserved"])
            self.assertEqual(
                summary["source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
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
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
            self.assertTrue(summary["audio_package_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertEqual(
                summary["next_recommended_issue"],
                "Stage B MIDI-to-solo targeted quality repair audio package source-context refresh",
            )

    def test_rejects_labeling_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=candidate_failure_labeling(
                        root,
                        quality_claim=True,
                    ),
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_rejects_labeling_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate_failure_labeling(root)
            source["source_schema_versions"]["quality_rubric_baseline"] = "wrong_schema"
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=source,
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_rejects_missing_labeling_schema_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate_failure_labeling(root)
            source["aggregate"]["outside_soloing_repair_schema_context_preserved"] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=source,
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_rejects_missing_outside_soloing_labeling_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate_failure_labeling(root)
            source["aggregate"]["outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=source,
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_rejects_missing_labeling_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate_failure_labeling(root)
            source["aggregate"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=source,
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_rejects_false_labeling_source_context_preserved_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = candidate_failure_labeling(root)
            source["aggregate"][
                "followup_objective_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairSweepError):
                build_targeted_quality_repair_sweep_report(
                    candidate_failure_labeling=source,
                    rubric_baseline=rubric_baseline(root),
                    output_dir=root / "sweep",
                    issue_number=1172,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_sweep")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_audio_package")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_targeted_quality_repair_sweep_v4")


if __name__ == "__main__":
    unittest.main()
