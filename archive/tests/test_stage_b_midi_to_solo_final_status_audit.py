from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY,
    NEXT_BOUNDARY,
    REQUIRED_README_SNIPPETS,
    SCHEMA_VERSION,
    StageBMidiToSoloFinalStatusAuditError,
    build_final_status_audit_report,
    validate_final_status_audit_report,
)
from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import (
    BOUNDARY as DELIVERY_BOUNDARY,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    LISTENING_GAP_SCHEMA_VERSION,
    NEXT_BOUNDARY as DELIVERY_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as DELIVERY_SCHEMA_VERSION,
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


def readme_text(*, missing_last: bool = False) -> str:
    snippets = REQUIRED_README_SNIPPETS[:-1] if missing_last else REQUIRED_README_SNIPPETS
    return "\n".join(snippets) + "\n"


def delivery_package(*, quality_claim: bool = False, cli_count: int = 3) -> dict:
    return {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "boundary": DELIVERY_BOUNDARY,
        "source_schema_versions": {
            "listening_review_quality_gap": LISTENING_GAP_SCHEMA_VERSION,
            "quality_gap_decision": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "current_evidence": CURRENT_EVIDENCE_SCHEMA_VERSION,
        },
        "delivery_package": {
            "runnable_cli_ready": True,
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
            "changed_ratio_repair_audio_evidence_ready": True,
            "outside_soloing_repair_evidence_ready": True,
            "outside_soloing_repair_source_context_preserved": True,
            "outside_soloing_repair_schema_context_preserved": True,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "cli_candidate_count": cli_count,
            "changed_ratio_repair_wav_count": 3,
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
            "listening_review_quality_gap_open": True,
            **SOURCE_CONTEXT,
        },
        "artifact_manifest": {
            "cli_repaired_midi_candidates": [{} for _ in range(cli_count)],
            "changed_ratio_repair_audio_candidates": [{} for _ in range(3)],
        },
        "readiness": {
            "mvp_delivery_package_completed": True,
            "raw_artifact_upload_required": False,
            "outside_soloing_repair_source_context_preserved": True,
            "outside_soloing_repair_schema_context_preserved": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": DELIVERY_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloFinalStatusAuditTest(unittest.TestCase):
    def test_audits_final_status_without_quality_claim(self) -> None:
        report = build_final_status_audit_report(
            delivery_package=delivery_package(),
            readme_text=readme_text(),
            output_dir=Path("outputs/final_status"),
            issue_number=1080,
        )
        summary = validate_final_status_audit_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_technical_mvp_complete=True,
            require_readme_reflected=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["final_status_audit_completed"])
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["source_delivery_package_schema_version"], DELIVERY_SCHEMA_VERSION)
        self.assertEqual(summary["source_listening_gap_schema_version"], LISTENING_GAP_SCHEMA_VERSION)
        self.assertEqual(
            summary["source_quality_gap_schema_version"],
            QUALITY_GAP_DECISION_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["technical_mvp_ready_for_local_review"])
        self.assertTrue(summary["readme_final_evidence_reflected"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["changed_ratio_repair_wav_count"], 3)
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(
            summary["outside_soloing_repair_source_objective_pitch_role_risk_count"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_before"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_after"],
            2,
        )
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["outside_soloing_repair_source_targeted"])
        self.assertTrue(summary["outside_soloing_repair_source_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertTrue(summary["listening_review_quality_gap_open"])
        self.assertFalse(summary["raw_artifact_upload_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo post-MVP musical quality iteration plan source-context refresh",
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["issue_number"], 1080)
        self.assertEqual(report["source_schema_versions"]["delivery_package"], DELIVERY_SCHEMA_VERSION)

    def test_rejects_final_status_schema_mismatch(self) -> None:
        report = build_final_status_audit_report(
            delivery_package=delivery_package(),
            readme_text=readme_text(),
            output_dir=Path("outputs/final_status"),
            issue_number=1164,
        )
        report["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            validate_final_status_audit_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_technical_mvp_complete=True,
                require_readme_reflected=True,
                require_no_quality_claim=True,
            )

    def test_rejects_delivery_source_schema_mismatch(self) -> None:
        source = delivery_package()
        source["source_schema_versions"]["current_evidence"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=source,
                readme_text=readme_text(),
                output_dir=Path("outputs/final_status"),
                issue_number=1164,
            )

    def test_rejects_false_schema_context_preserved_flag(self) -> None:
        source = delivery_package()
        source["delivery_package"]["outside_soloing_repair_schema_context_preserved"] = False
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=source,
                readme_text=readme_text(),
                output_dir=Path("outputs/final_status"),
                issue_number=1164,
            )

    def test_rejects_missing_readme_snippet(self) -> None:
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=delivery_package(),
                readme_text=readme_text(missing_last=True),
                output_dir=Path("outputs/final_status"),
                issue_number=742,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=delivery_package(quality_claim=True),
                readme_text=readme_text(),
                output_dir=Path("outputs/final_status"),
                issue_number=742,
            )

    def test_rejects_low_cli_candidate_count(self) -> None:
        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=delivery_package(cli_count=2),
                readme_text=readme_text(),
                output_dir=Path("outputs/final_status"),
                issue_number=742,
            )

    def test_rejects_false_source_context_preserved_flag(self) -> None:
        source = delivery_package()
        source["delivery_package"][
            "repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False

        with self.assertRaises(StageBMidiToSoloFinalStatusAuditError):
            build_final_status_audit_report(
                delivery_package=source,
                readme_text=readme_text(),
                output_dir=Path("outputs/final_status"),
                issue_number=1080,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_final_status_audit")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_post_mvp_quality_iteration_plan")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_final_status_audit_v4")


if __name__ == "__main__":
    unittest.main()
