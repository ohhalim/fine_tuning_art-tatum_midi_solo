from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.guard_music_transformer_solo_yield_residual_aware_listening_input import (
    STATUS_SYNC_BOUNDARY,
    SoloYieldResidualAwareListeningInputGuardError,
    build_guard_report,
    validate_guard_report,
)


def source_package(*, validated_input: bool = False, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_final_review_package_v1",
        "aggregate": {
            "candidate_count": 2,
            "midi_count": 2,
            "wav_count": 2,
            "quality_proxy_pass_count": 1,
            "quality_proxy_fail_count": 1,
            "major_label_counts": {"low_tension_color": 1},
            "watch_label_counts": {"dead_air_watch": 1},
        },
        "candidate_handoff": [
            {"review_index": 1},
            {"review_index": 2},
        ],
        "readiness": {
            "residual_aware_final_review_package_ready": True,
            "review_input_template_written": True,
            "validated_listening_input_present": validated_input,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_listening_input_guard",
            "critical_user_input_required": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareListeningInputGuardTest(unittest.TestCase):
    def test_blocks_preference_fill_when_input_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_guard_report(
                source_package(),
                output_dir=Path(raw_temp) / "guard",
                issue_number=1390,
            )
        summary = validate_guard_report(
            report,
            expected_next_boundary=STATUS_SYNC_BOUNDARY,
            require_pending_input=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["listening_review_completed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertEqual(summary["review_item_count"], 2)

    def test_validation_rejects_unexpected_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_guard_report(
                source_package(),
                output_dir=Path(raw_temp) / "guard",
                issue_number=1390,
            )

        with self.assertRaises(SoloYieldResidualAwareListeningInputGuardError):
            validate_guard_report(
                report,
                expected_next_boundary="other",
                require_pending_input=True,
                require_no_quality_claim=True,
            )

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareListeningInputGuardError):
                build_guard_report(
                    source_package(quality_claim=True),
                    output_dir=Path(raw_temp) / "guard",
                    issue_number=1390,
                )


if __name__ == "__main__":
    unittest.main()
