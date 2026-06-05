from __future__ import annotations

import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Sequence

from scripts.render_stage_b_midi_to_solo_phrase_bank_audio import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankAudioRenderError,
    build_audio_render_report,
    validate_audio_render_report,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
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


def phrase_bank_report(root: Path, *, quality_claim: bool = False) -> dict:
    top_candidates = [
        {
            "rank": index,
            "mode": "data_motif_rhythm_phrase_variation",
            "sample_index": index,
            "sample_seed": 632 + index,
            "export_midi_path": touch_file(root, f"midi/rank_{index}.mid"),
            "generation_source": "phrase_bank_data_motif_retrieval",
            "export_contract_gate_passed": True,
            "exported_metrics": {
                "note_count": 64,
                "unique_pitch_count": 20 + index,
                "max_simultaneous_notes": 1,
                "dead_air_ratio": 0.58,
                "phrase_coverage_ratio": 1.0,
            },
        }
        for index in range(1, 4)
    ]
    return {
        "boundary": SOURCE_BOUNDARY,
        "summary": {
            "candidate_count": 9,
            "exported_candidate_count": 3,
            "exported_qualified_candidate_count": 3,
        },
        "top_candidates": top_candidates,
        "readiness": {
            "phrase_bank_template_extracted": True,
            "phrase_bank_retrieval_baseline_completed": True,
            "ranked_midi_candidates_exported": True,
            "midi_to_solo_mvp_claimed": False,
            "phrase_bank_musical_quality_claimed": quality_claim,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
        },
        "decision": {
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseBankAudioRenderTest(unittest.TestCase):
    def test_renders_phrase_bank_candidates_to_wav_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            report = build_audio_render_report(
                phrase_bank_report(root),
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
                require_phrase_bank_audio_path=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["render_attempted"])
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertTrue(summary["phrase_bank_ranked_audio_render_completed"])
        self.assertTrue(summary["phrase_bank_listening_review_package_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_phrase_bank_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            with self.assertRaises(StageBMidiToSoloPhraseBankAudioRenderError):
                build_audio_render_report(
                    phrase_bank_report(root, quality_claim=True),
                    output_dir=root / "out",
                    renderer_path=renderer,
                    soundfont_path=soundfont,
                    sample_rate=44100,
                    expected_file_count=3,
                    runner=fake_runner,
                )

    def test_rejects_missing_phrase_bank_midi_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = touch_file(root, "bin/fluidsynth")
            soundfont = touch_file(root, "sf/general.sf2")
            source = phrase_bank_report(root)
            source["top_candidates"][0]["export_midi_path"] = "missing.mid"
            with self.assertRaises(StageBMidiToSoloPhraseBankAudioRenderError):
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
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_audio_render_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_listening_review_package")


if __name__ == "__main__":
    unittest.main()
