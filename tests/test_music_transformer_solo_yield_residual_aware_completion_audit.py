from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_music_transformer_solo_yield_residual_aware_completion import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareCompletionAuditError,
    build_completion_audit_report,
    validate_completion_audit_report,
)


def pending_report(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_listening_review_pending_v1",
        "output_dir": "outputs/pending",
        "aggregate": {
            "candidate_count": 2,
            "midi_count": 2,
            "wav_count": 2,
            "pending_candidate_count": 2,
            "quality_proxy_pass_count": 1,
            "quality_proxy_fail_count": 1,
            "major_label_counts": {"low_tension_color": 1},
            "watch_label_counts": {"dead_air_watch": 1},
        },
        "review_input": {
            "path": "outputs/review_input.json",
            "review_status": "pending",
        },
        "readiness": {
            "residual_aware_listening_review_pending_recorded": True,
            "local_mvp_handoff_ready": True,
            "review_input_template_pending": True,
            "manual_review_required_for_quality_claim": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_completion_audit",
            "critical_user_input_required": False,
        },
    }


def handoff_freeze(*, checksum_mismatch_count: int = 0) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze_v1",
        "output_dir": "outputs/handoff",
        "aggregate": {
            "candidate_count": 2,
            "midi_count": 2,
            "wav_count": 2,
            "missing_file_count": 0,
            "checksum_mismatch_count": checksum_mismatch_count,
            "raw_artifact_upload_required": False,
        },
        "readiness": {
            "residual_aware_mvp_handoff_freeze_completed": True,
            "local_candidate_artifacts_verified": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareCompletionAuditTest(unittest.TestCase):
    def test_audits_technical_completion_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_completion_audit_report(
                pending_report=pending_report(),
                handoff_freeze_report=handoff_freeze(),
                output_dir=Path(raw_temp) / "completion",
                issue_number=1400,
            )
        summary = validate_completion_audit_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_technical_complete=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["local_review_ready"])
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["pending_candidate_count"], 2)
        self.assertEqual(summary["checksum_mismatch_count"], 0)
        self.assertTrue(summary["manual_review_required_for_quality_claim"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["raw_artifact_upload_required"])

    def test_rejects_quality_claim_in_pending_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareCompletionAuditError):
                build_completion_audit_report(
                    pending_report=pending_report(quality_claim=True),
                    handoff_freeze_report=handoff_freeze(),
                    output_dir=Path(raw_temp) / "completion",
                    issue_number=1400,
                )

    def test_rejects_handoff_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareCompletionAuditError):
                build_completion_audit_report(
                    pending_report=pending_report(),
                    handoff_freeze_report=handoff_freeze(checksum_mismatch_count=1),
                    output_dir=Path(raw_temp) / "completion",
                    issue_number=1400,
                )


if __name__ == "__main__":
    unittest.main()
