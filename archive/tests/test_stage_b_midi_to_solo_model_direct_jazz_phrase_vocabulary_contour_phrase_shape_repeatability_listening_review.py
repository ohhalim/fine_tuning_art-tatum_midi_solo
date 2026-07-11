from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package import (
    BOUNDARY as AUDIO_BOUNDARY,
)
from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError,
    build_listening_review_report,
    validate_listening_review_report,
)


def fake_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + (b"\x00" * 128))


def audio_package(root: Path, *, quality_claim: bool = False, technical_valid: bool = True) -> dict:
    rendered = []
    for rank in range(1, 7):
        wav_path = root / f"repeatability_seed_{rank:02d}.wav"
        fake_wav(wav_path)
        rendered.append(
            {
                "rank": rank,
                "midi_path": str(root / f"repeatability_seed_{rank:02d}.mid"),
                "midi_sha256": str(rank) * 64,
                "wav_file": {
                    "path": str(wav_path),
                    "exists": True,
                    "duration_seconds": 18.9 + (rank * 0.01),
                    "sample_rate": 44100,
                    "size_bytes": wav_path.stat().st_size,
                    "sha256": str(rank) * 64,
                },
            }
        )
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package_v1",
        "audio_package_boundary": {
            "boundary": AUDIO_BOUNDARY,
            "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_consolidation",
            "candidate_count": 6,
            "rendered_audio_file_count": 6,
            "technical_wav_validation": technical_valid,
            "listening_review_completed": False,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "rendered_audio_files": rendered,
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewTest(
    unittest.TestCase
):
    def test_builds_pending_repeatability_listening_review_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_listening_review_report(
                audio_package(root),
                output_dir=root / "review",
                expected_file_count=6,
            )
            summary = validate_listening_review_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_file_count=6,
                require_pending_review=True,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["candidate_count"], 6)
        self.assertEqual(summary["rendered_audio_file_count"], 6)
        self.assertTrue(summary["review_input_template_written"])
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["pending_status_field_count"], 4)
        self.assertEqual(summary["pending_candidate_decision_count"], 6)
        self.assertEqual(summary["pending_candidate_field_count"], 18)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(
                StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError
            ):
                build_listening_review_report(
                    audio_package(root, quality_claim=True),
                    output_dir=root / "review",
                    expected_file_count=6,
                )

    def test_rejects_missing_technical_wav_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(
                StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityListeningReviewError
            ):
                build_listening_review_report(
                    audio_package(root, technical_valid=False),
                    output_dir=root / "review",
                    expected_file_count=6,
                )


if __name__ == "__main__":
    unittest.main()
