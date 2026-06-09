from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.render_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPitchContourChangedRatioRepairAudioError,
    build_audio_render_report,
    validate_audio_render_report,
)
from scripts.run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 2 * 1200)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(str(command[3]))
    write_wav(wav_path)
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"midi")
    return str(path)


def source_report(root: Path, *, quality_claim: bool = False) -> dict:
    rows = []
    for index in range(1, 4):
        rows.append(
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 698 + index,
                "source_midi_path": touch_file(root, f"source/rank_{index}.mid"),
                "repaired_midi_path": touch_file(root, f"repaired/rank_{index}.mid"),
                "source_metrics": {
                    "max_interval": 62,
                },
                "repaired_metrics": {
                    "note_count": 46,
                    "unique_pitch_count": 20 + index,
                    "max_simultaneous_notes": 1,
                    "dead_air_ratio": 0.0,
                    "max_interval": 8 + index,
                },
                "pitch_repair_stats": {
                    "pitch_changed_ratio": 0.4,
                    "max_pitch_shift_abs": 48,
                },
                "candidate_repair_passed": True,
            }
        )
    return {
        "boundary": SOURCE_BOUNDARY,
        "summary": {
            "repaired_pass_count": 3,
            "repaired_candidate_count": 3,
            "source_max_interval": 62,
            "repaired_max_interval": 11,
            "target_max_interval": 12,
            "repaired_dead_air_max": 0.0,
            "repaired_max_pitch_changed_ratio": 0.4,
            "max_pitch_changed_ratio": 0.5,
            "changed_ratio_repair_passed": True,
        },
        "candidate_repairs": rows,
        "readiness": {
            "changed_ratio_repair_probe_completed": True,
            "repaired_ranked_midi_written": True,
            "changed_ratio_repair_passed": True,
            "changed_ratio_repair_audio_render_required": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPitchContourChangedRatioRepairAudioTest(unittest.TestCase):
    def test_renders_changed_ratio_repaired_midi_to_wav_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            report = build_audio_render_report(
                source_report(root),
                output_dir=root / "out",
                renderer_path=renderer,
                soundfont_path=soundfont,
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_audio_render_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_audio_package_completed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["render_attempted"])
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertTrue(summary["changed_ratio_repair_audio_package_completed"])
        self.assertEqual(summary["max_repaired_interval"], 11)
        self.assertEqual(summary["repaired_dead_air_max"], 0.0)
        self.assertLessEqual(summary["max_repaired_pitch_changed_ratio"], 0.5)
        self.assertTrue(summary["audio_review_required"])
        self.assertFalse(summary["audio_rendered_quality_claimed"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            with self.assertRaises(
                StageBMidiToSoloPitchContourChangedRatioRepairAudioError
            ):
                build_audio_render_report(
                    source_report(root, quality_claim=True),
                    output_dir=root / "out",
                    renderer_path=renderer,
                    soundfont_path=soundfont,
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_rejects_missing_repaired_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            source = source_report(root)
            source["candidate_repairs"][0]["repaired_midi_path"] = "missing.mid"
            with self.assertRaises(
                StageBMidiToSoloPitchContourChangedRatioRepairAudioError
            ):
                build_audio_render_report(
                    source,
                    output_dir=root / "out",
                    renderer_path=renderer,
                    soundfont_path=soundfont,
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package",
        )


if __name__ == "__main__":
    unittest.main()
