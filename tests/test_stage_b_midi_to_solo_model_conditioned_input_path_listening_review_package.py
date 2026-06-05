from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathListeningReviewPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
)
from scripts.consolidate_stage_b_midi_to_solo_model_conditioned_input_path_replacement import (
    BOUNDARY as REPLACEMENT_BOUNDARY,
    NEXT_BOUNDARY as REPLACEMENT_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_audio import (
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
)


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return str(path)


def replacement_report() -> dict:
    return {
        "boundary": REPLACEMENT_BOUNDARY,
        "readiness": {
            "model_conditioned_input_path_replacement_consolidated": True,
            "model_conditioned_input_to_ranked_midi_completed": True,
            "model_conditioned_input_to_ranked_wav_completed": True,
            "fallback_replacement_technical_path_ready": True,
            "listening_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": REPLACEMENT_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def audio_render_report(root: Path, *, quality_claim: bool = False) -> dict:
    files = [
        {
            "rank": index,
            "sample_index": index,
            "sample_seed": 490 + index,
            "source_midi_path": touch_file(root, f"midi/rank_{index}.mid"),
            "source_score": 10.0 + index,
            "source_note_count": 24,
            "source_unique_pitch_count": 12,
            "source_chord_tone_ratio": 0.7,
            "source_dead_air_ratio": 0.5,
            "wav_file": {
                "path": touch_file(root, f"audio/rank_{index}.wav"),
                "exists": True,
                "duration_seconds": 1.0 + index,
                "sample_rate": 44100,
                "size_bytes": 1000,
                "sha256": f"sha-{index}",
            },
        }
        for index in range(1, 4)
    ]
    return {
        "audio_render_boundary": {
            "boundary": AUDIO_RENDER_BOUNDARY,
            "technical_wav_validation": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "rendered_audio_files": files,
    }


class StageBMidiToSoloModelConditionedInputPathListeningReviewPackageTest(unittest.TestCase):
    def test_builds_pending_listening_review_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_package_report(
                replacement_report=replacement_report(),
                audio_render_report=audio_render_report(root),
                output_dir=root / "out",
                issue_number=630,
                expected_count=3,
            )
            summary = validate_listening_review_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_review_item_count=3,
                require_package_ready=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["listening_review_package_ready"])
        self.assertEqual(summary["review_item_count"], 3)
        self.assertFalse(summary["validated_review_input"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(len(summary["wav_paths"]), 3)

    def test_rejects_audio_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathListeningReviewPackageError):
                build_listening_review_package_report(
                    replacement_report=replacement_report(),
                    audio_render_report=audio_render_report(root, quality_claim=True),
                    output_dir=root / "out",
                    issue_number=630,
                    expected_count=3,
                )

    def test_rejects_missing_review_item_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = audio_render_report(root)
            audio["rendered_audio_files"][0]["wav_file"]["path"] = "missing.wav"
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathListeningReviewPackageError):
                build_listening_review_package_report(
                    replacement_report=replacement_report(),
                    audio_render_report=audio,
                    output_dir=root / "out",
                    issue_number=630,
                    expected_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard")


if __name__ == "__main__":
    unittest.main()
