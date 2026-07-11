from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPitchContourChangedRatioRepairListeningReviewPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return str(path)


def audio_package_report(root: Path, *, quality_claim: bool = False) -> dict:
    files = []
    for index in range(1, 4):
        files.append(
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 698 + index,
                "repaired_midi_path": touch_file(root, f"midi/rank_{index}.mid"),
                "repaired_dead_air_ratio": 0.0,
                "repaired_max_interval": 8 + index,
                "repaired_unique_pitch_count": 20 + index,
                "pitch_changed_ratio": 0.4,
                "wav_file": {
                    "path": touch_file(root, f"audio/rank_{index}.wav"),
                    "exists": True,
                    "duration_seconds": 18.0 + index,
                    "sample_rate": 44100,
                    "size_bytes": 1000,
                    "sha256": f"sha-{index}",
                },
            }
        )
    return {
        "summary": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "repaired_dead_air_max": 0.0,
            "max_repaired_interval": 11,
            "min_repaired_unique_pitch_count": 21,
            "max_repaired_pitch_changed_ratio": 0.4,
            "target_max_pitch_changed_ratio": 0.5,
            "audio_review_required": True,
        },
        "audio_render_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "changed_ratio_repair_audio_package_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "rendered_audio_files": files,
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPitchContourChangedRatioRepairListeningReviewPackageTest(
    unittest.TestCase
):
    def test_builds_pending_listening_review_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_package_report(
                audio_package_report=audio_package_report(root),
                output_dir=root / "out",
                issue_number=722,
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
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["max_repaired_interval"], 11)
        self.assertLessEqual(summary["max_repaired_pitch_changed_ratio"], 0.5)
        self.assertTrue(summary["audio_review_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(len(summary["wav_paths"]), 3)

    def test_rejects_audio_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(
                StageBMidiToSoloPitchContourChangedRatioRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=audio_package_report(root, quality_claim=True),
                    output_dir=root / "out",
                    issue_number=722,
                    expected_count=3,
                )

    def test_rejects_missing_review_item_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = audio_package_report(root)
            audio["rendered_audio_files"][0]["wav_file"]["path"] = "missing.wav"
            with self.assertRaises(
                StageBMidiToSoloPitchContourChangedRatioRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=audio,
                    output_dir=root / "out",
                    issue_number=722,
                    expected_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard",
        )


if __name__ == "__main__":
    unittest.main()
