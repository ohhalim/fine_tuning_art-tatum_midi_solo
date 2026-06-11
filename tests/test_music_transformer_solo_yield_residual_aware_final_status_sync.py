from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sync_music_transformer_solo_yield_residual_aware_final_status import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareFinalStatusSyncError,
    build_final_status_sync_report,
    validate_final_status_sync_report,
)


def completion_audit(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_completion_audit_v1",
        "output_dir": "outputs/completion",
        "aggregate": {
            "candidate_count": 8,
            "midi_count": 8,
            "wav_count": 8,
            "pending_candidate_count": 8,
            "quality_proxy_pass_count": 6,
            "quality_proxy_fail_count": 2,
            "major_label_counts": {"low_tension_color": 2},
            "watch_label_counts": {"dead_air_watch": 3},
            "missing_file_count": 0,
            "checksum_mismatch_count": 0,
            "raw_artifact_upload_required": False,
        },
        "readiness": {
            "residual_aware_completion_audit_completed": True,
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "manual_review_required_for_quality_claim": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_final_status_sync",
            "critical_user_input_required": False,
        },
    }


def readme_text(*, missing_completion: bool = False) -> str:
    lines = [
        "residual-aware completion audit: technical MVP complete `true`, local review ready `true`",
        "completion audit doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`",
        "최신 review package: MIDI `8`, WAV `8`",
        "objective rubric: pass/fail `6 / 2`",
        "남은 major label: `low_tension_color=2`",
        "남은 watch label: `dead_air_watch=3`",
        "validated listening input: `false`",
        "musical quality claim: `false`",
    ]
    if missing_completion:
        lines.pop(0)
    return "\n".join(lines)


def current_status_text() -> str:
    return "\n".join(
        [
            "- current issue: Issue #1402, Stage B MIDI-to-solo residual-aware final status sync",
            "- residual-aware completion audit technical MVP complete: `true`",
            "- residual-aware completion audit local review ready: `true`",
            "- residual-aware completion audit quality claim: `false`",
            "- residual-aware completion audit next boundary: `music_transformer_solo_yield_residual_aware_final_status_sync`",
            "- residual-aware final status sync candidate count: `8`",
            "- residual-aware final status sync MIDI/WAV: `8 / 8`",
            "- stable jazz solo quality: `not_proven`",
            "- human listening preference input: `false`",
        ]
    )


class MusicTransformerSoloYieldResidualAwareFinalStatusSyncTest(unittest.TestCase):
    def test_syncs_final_status_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_final_status_sync_report(
                completion_audit=completion_audit(),
                readme_text=readme_text(),
                current_status_text=current_status_text(),
                output_dir=Path(raw_temp) / "sync",
                issue_number=1402,
            )
        summary = validate_final_status_sync_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_final_status_synced=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["residual_aware_final_status_synced"])
        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["local_review_ready"])
        self.assertEqual(summary["candidate_count"], 8)
        self.assertEqual(summary["pending_candidate_count"], 8)
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_validation_rejects_missing_readme_sync(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_final_status_sync_report(
                completion_audit=completion_audit(),
                readme_text=readme_text(missing_completion=True),
                current_status_text=current_status_text(),
                output_dir=Path(raw_temp) / "sync",
                issue_number=1402,
            )

        with self.assertRaises(SoloYieldResidualAwareFinalStatusSyncError):
            validate_final_status_sync_report(
                report,
                expected_next_boundary=NEXT_BOUNDARY,
                require_final_status_synced=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim_in_completion_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareFinalStatusSyncError):
                build_final_status_sync_report(
                    completion_audit=completion_audit(quality_claim=True),
                    readme_text=readme_text(),
                    current_status_text=current_status_text(),
                    output_dir=Path(raw_temp) / "sync",
                    issue_number=1402,
                )


if __name__ == "__main__":
    unittest.main()
