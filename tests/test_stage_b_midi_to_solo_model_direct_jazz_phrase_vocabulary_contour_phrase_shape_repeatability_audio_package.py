from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError,
    build_audio_package_report,
    validate_audio_package_report,
)
from scripts.consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability import (
    BOUNDARY as CONSOLIDATION_BOUNDARY,
)


def fake_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00")


def write_wav(path: Path, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00\x00\x00" * sample_rate)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    output_index = list(command).index("-F") + 1
    write_wav(Path(command[output_index]))
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def consolidation(root: Path, *, support: bool = True, quality_claim: bool = False) -> dict:
    midi_paths = [root / f"repeatability_seed_{index:02d}.mid" for index in range(1, 7)]
    for path in midi_paths:
        fake_midi(path)
    return {
        "schema_version": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation_v1",
        "boundary": CONSOLIDATION_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_sweep",
        "evidence_summary": {
            "sample_count": 6,
            "generated_midi_file_count": 6,
            "qualified_candidate_count": 6,
            "objective_clean_pass_rate": 1.0,
            "current_analysis_flag_count": 0,
            "overlap_detected_count": 0,
            "distinct_density_pattern_count": 6,
            "max_abs_interval_max": 12,
            "max_small_interval_ratio_le4": 0.1765,
            "generated_midi_paths": [str(path) for path in midi_paths],
        },
        "consolidation_result": {
            "objective_repeatability_support": support,
            "additional_repair_required": not support,
            "audio_review_package_required": support,
            "support_scope": "objective_midi_only",
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": CONSOLIDATION_BOUNDARY,
            "repeatability_consolidation_completed": True,
            "objective_repeatability_support": support,
            "audio_review_package_required": support,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": CONSOLIDATION_BOUNDARY,
            "next_boundary": BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageTest(
    unittest.TestCase
):
    def test_builds_repeatability_audio_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            report = build_audio_package_report(
                consolidation(root),
                output_dir=root / "audio_package",
                renderer_path=str(renderer),
                soundfont_path=str(soundfont),
                sample_rate=44100,
                expected_file_count=6,
                runner=fake_runner,
            )
            summary = validate_audio_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_file_count=6,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["candidate_count"], 6)
        self.assertEqual(summary["rendered_audio_file_count"], 6)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertFalse(summary["listening_review_completed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(len(summary["wav_paths"]), 6)

    def test_rejects_unsupported_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError):
                build_audio_package_report(
                    consolidation(root, support=False),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            soundfont.write_bytes(b"sf2")
            with self.assertRaises(StageBMidiToSoloModelDirectJazzPhraseVocabularyContourPhraseShapeRepeatabilityAudioPackageError):
                build_audio_package_report(
                    consolidation(root, quality_claim=True),
                    output_dir=root / "audio_package",
                    renderer_path=str(renderer),
                    soundfont_path=str(soundfont),
                    sample_rate=44100,
                    expected_file_count=6,
                    runner=fake_runner,
                )


if __name__ == "__main__":
    unittest.main()
