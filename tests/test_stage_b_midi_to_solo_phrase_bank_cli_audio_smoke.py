from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankCliAudioSmokeError,
    build_audio_smoke_report,
    validate_audio_smoke_report,
)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 2 * 1000)


def fake_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wav_path = Path(str(command[3]))
    write_wav(wav_path)
    return subprocess.CompletedProcess(list(command), 0, stdout="rendered", stderr="")


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"midi")
    return str(path)


def smoke_report(root: Path, *, quality_claim: bool = False) -> dict:
    candidates = []
    for rank in range(1, 4):
        candidates.append(
            {
                "rank": rank,
                "sample_seed": 630 + rank,
                "repaired_midi_path": touch_file(root, f"midi/rank_{rank}.mid"),
                "objective_supported": True,
                "note_count": 96,
                "unique_pitch_count": 20 + rank,
                "max_simultaneous_notes": 1,
                "dead_air_ratio": 0.2,
                "phrase_coverage_ratio": 1.0,
            }
        )
    return {
        "boundary": SOURCE_BOUNDARY,
        "input": {
            "midi_path": touch_file(root, "input/source.mid"),
            "explicit_input_used": True,
        },
        "candidate_manifest": candidates,
        "readiness": {
            "user_input_smoke_completed": True,
            "explicit_input_path_used": True,
            "ranked_repaired_midi_exported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "phrase_bank_musical_quality_claimed": False,
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


class StageBMidiToSoloPhraseBankCliAudioSmokeTest(unittest.TestCase):
    def test_renders_cli_smoke_candidates_to_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            report = build_audio_smoke_report(
                smoke_report(root),
                output_dir=root / "out",
                renderer_path=renderer,
                soundfont_path=soundfont,
                sample_rate=44100,
                expected_file_count=3,
                runner=fake_runner,
            )
            summary = validate_audio_smoke_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_file_count=3,
                expected_sample_rate=44100,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertTrue(summary["cli_user_input_audio_render_completed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")

            with self.assertRaises(StageBMidiToSoloPhraseBankCliAudioSmokeError):
                build_audio_smoke_report(
                    smoke_report(root, quality_claim=True),
                    output_dir=root / "out",
                    renderer_path=renderer,
                    soundfont_path=soundfont,
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package")


if __name__ == "__main__":
    unittest.main()
