from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.mark_music_transformer_solo_yield_residual_aware_listening_review_pending import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareListeningReviewPendingError,
    build_pending_report,
    validate_pending_report,
)


def review_input_template(path: Path, *, filled: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": "music_transformer_solo_yield_residual_aware_review_input_v1",
        "review_status": "pending",
        "overall_decision": "pending",
        "candidates": [
            {
                "review_index": 1,
                "decision": "keep" if filled else "pending",
                "usable_as_jazz_solo_phrase": True if filled else None,
                "primary_failure": None,
            },
            {
                "review_index": 2,
                "decision": "pending",
                "usable_as_jazz_solo_phrase": None,
                "primary_failure": None,
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def handoff_freeze(root: Path, *, quality_claim: bool = False, filled_input: bool = False) -> dict:
    template_path = root / "review_input_template.json"
    review_input_template(template_path, filled=filled_input)
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze_v1",
        "output_dir": "outputs/handoff",
        "artifact_paths": {
            "review_input_template_json": str(template_path),
        },
        "aggregate": {
            "candidate_count": 2,
            "midi_count": 2,
            "wav_count": 2,
            "quality_proxy_pass_count": 1,
            "quality_proxy_fail_count": 1,
            "major_label_counts": {"low_tension_color": 1},
            "watch_label_counts": {"dead_air_watch": 1},
            "missing_file_count": 0,
            "checksum_mismatch_count": 0,
            "raw_artifact_upload_required": False,
        },
        "readiness": {
            "residual_aware_mvp_handoff_freeze_completed": True,
            "local_candidate_artifacts_verified": True,
            "review_input_template_available": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_listening_review_pending",
            "critical_user_input_required": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareListeningReviewPendingTest(unittest.TestCase):
    def test_records_pending_boundary_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            report = build_pending_report(
                handoff_freeze_report=handoff_freeze(root),
                output_dir=root / "pending",
                issue_number=1398,
            )
        summary = validate_pending_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_pending_review=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["local_mvp_handoff_ready"])
        self.assertTrue(summary["review_input_template_pending"])
        self.assertTrue(summary["manual_review_required_for_quality_claim"])
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["pending_candidate_count"], 2)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["listening_review_completed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_filled_review_input_template(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualAwareListeningReviewPendingError):
                build_pending_report(
                    handoff_freeze_report=handoff_freeze(root, filled_input=True),
                    output_dir=root / "pending",
                    issue_number=1398,
                )

    def test_rejects_quality_claim_in_handoff_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualAwareListeningReviewPendingError):
                build_pending_report(
                    handoff_freeze_report=handoff_freeze(root, quality_claim=True),
                    output_dir=root / "pending",
                    issue_number=1398,
                )


if __name__ == "__main__":
    unittest.main()
