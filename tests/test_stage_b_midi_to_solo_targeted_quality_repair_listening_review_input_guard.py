from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_targeted_quality_repair_listening_review_package import (
    BOUNDARY as SOURCE_BOUNDARY,
    EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_PACKAGE_SCHEMA_VERSION,
)
from scripts.guard_stage_b_midi_to_solo_targeted_quality_repair_listening_review_input import (
    BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloTargetedQualityRepairListeningInputGuardError,
    build_listening_review_input_guard_report,
    validate_listening_review_input_guard_report,
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


def source_package(*, quality_claim: bool = False, validated_input: bool = False) -> dict:
    review_items = [
        {
            "candidate_index": index,
            "source": "unit_source",
            "rank": index,
            "midi_path": f"/tmp/candidate_{index}.mid",
            "wav_path": f"/tmp/candidate_{index}.wav",
            "duration_seconds": 0.1,
            "sample_rate": 44100,
            "size_bytes": 100,
            "sha256": "abc",
            "repaired_failure_labels": ["songlike_melody_not_soloing"]
            if index < 6
            else [],
            "review_status": "pending",
        }
        for index in range(1, 7)
    ]
    return {
        "schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "boundary": SOURCE_BOUNDARY,
        "source_schema_versions": dict(EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS),
        "source_summary": {
            "source_targeted_quality_repair_audio_package_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS[
                    "targeted_quality_repair_audio_package"
                ]
            ),
            "source_targeted_quality_repair_sweep_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["targeted_quality_repair_sweep"]
            ),
            "source_candidate_failure_labeling_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["candidate_failure_labeling"]
            ),
            "source_quality_rubric_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["quality_rubric_baseline"]
            ),
            "source_post_mvp_plan_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["post_mvp_quality_iteration_plan"]
            ),
            "source_final_status_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["final_status_audit"]
            ),
            "source_delivery_package_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["delivery_package"]
            ),
            "source_listening_gap_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["listening_review_quality_gap"]
            ),
            "source_quality_gap_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["quality_gap_decision"]
            ),
            "source_current_evidence_schema_version": (
                EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS["current_evidence"]
            ),
            "rendered_audio_file_count": 6,
            "technical_wav_validation": True,
            "sample_rate": 44100,
            "duration_min_seconds": 0.1,
            "duration_max_seconds": 0.1,
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
        "review_package": {
            "package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "required_input_fields": [
                "candidate_index",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": review_items,
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "human_review_required_now": False,
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


class StageBMidiToSoloTargetedQualityRepairListeningInputGuardTest(unittest.TestCase):
    def test_blocks_preference_fill_when_review_input_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_input_guard_report(
                source_package(),
                output_dir=root / "guard",
                issue_number=1178,
            )
            summary = validate_listening_review_input_guard_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
                require_guard_completed=True,
                require_preference_blocked=True,
                require_pending_input=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1178)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                report["source_schema_versions"][
                    "targeted_quality_repair_listening_review_package"
                ],
                SOURCE_PACKAGE_SCHEMA_VERSION,
            )
            for key, expected in EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS.items():
                self.assertEqual(report["source_schema_versions"][key], expected)
            self.assertEqual(
                summary["source_targeted_quality_repair_listening_review_package_schema_version"],
                SOURCE_PACKAGE_SCHEMA_VERSION,
            )
            for key, expected in EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS.items():
                summary_key = "source_" + {
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
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertEqual(summary["review_item_count"], 6)
            self.assertEqual(summary["required_input_field_count"], 4)
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
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertEqual(
                summary["next_recommended_issue"],
                "Stage B MIDI-to-solo targeted quality repair objective-only next decision source-context refresh",
            )

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source_package(quality_claim=True),
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_summary"]["repaired_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_source_risk_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_summary"]["source_outside_soloing_repair_source_pitch_role_risk_delta"] = 1
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_missing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_summary"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_false_source_context_preserved_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_summary"][
                "followup_objective_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_source_schema_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_schema_versions"]["targeted_quality_repair_audio_package"] = (
                "wrong_schema"
            )
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_rejects_missing_schema_context_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            source["source_summary"]["source_outside_soloing_repair_schema_context_preserved"] = (
                False
            )
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=1178,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_listening_review_input_guard")
        self.assertEqual(OBJECTIVE_NEXT_BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_objective_only_next_decision")
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_targeted_quality_repair_listening_review_input_guard_v5",
        )


if __name__ == "__main__":
    unittest.main()
