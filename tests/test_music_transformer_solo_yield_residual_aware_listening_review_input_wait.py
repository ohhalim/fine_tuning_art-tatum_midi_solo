from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.mark_music_transformer_solo_yield_residual_aware_listening_review_input_wait import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareListeningReviewInputWaitError,
    build_input_wait_report,
    validate_input_wait_report,
)


def final_status_sync(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_final_status_sync_v1",
        "output_dir": "outputs/final_status",
        "aggregate": {
            "candidate_count": 8,
            "midi_count": 8,
            "wav_count": 8,
            "pending_candidate_count": 8,
            "quality_proxy_pass_count": 6,
            "quality_proxy_fail_count": 2,
            "major_label_counts": {"low_tension_color": 2},
            "watch_label_counts": {"dead_air_watch": 3},
        },
        "readiness": {
            "residual_aware_final_status_sync_completed": True,
            "residual_aware_final_status_synced": True,
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_listening_review_input_wait",
            "critical_user_input_required": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareListeningReviewInputWaitTest(unittest.TestCase):
    def test_records_wait_state_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_input_wait_report(
                final_status_sync_report=final_status_sync(),
                output_dir=Path(raw_temp) / "wait",
                issue_number=1404,
            )
        summary = validate_input_wait_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_wait_recorded=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["final_status_synced"])
        self.assertTrue(summary["local_review_ready"])
        self.assertTrue(summary["user_listening_input_required_for_quality_claim"])
        self.assertTrue(summary["automated_quality_claim_blocked"])
        self.assertEqual(summary["candidate_count"], 8)
        self.assertEqual(summary["pending_candidate_count"], 8)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_quality_claim_in_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareListeningReviewInputWaitError):
                build_input_wait_report(
                    final_status_sync_report=final_status_sync(quality_claim=True),
                    output_dir=Path(raw_temp) / "wait",
                    issue_number=1404,
                )


if __name__ == "__main__":
    unittest.main()
