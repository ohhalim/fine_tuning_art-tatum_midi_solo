from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.analyze_stage_b_midi_to_solo_model_direct_songlike_rejection import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError,
    analyze_midi_candidate,
    build_songlike_rejection_analysis_report,
    validate_songlike_rejection_analysis_report,
)


def write_songlike_midi(path: Path, pitches: list[int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    bar_patterns = [
        [0.0, 0.125, 0.5, 0.875],
        [0.125, 0.25, 0.625, 1.0],
        [0.0, 0.125, 0.375, 0.75],
        [0.25, 0.375, 0.75, 1.125],
    ]
    duration_pattern = [0.125, 0.375, 0.375, 0.375]
    index = 0
    for bar in range(8):
        bar_start = bar * 2.0
        for note_index, offset in enumerate(bar_patterns[bar % 4]):
            pitch = pitches[index % len(pitches)]
            start = bar_start + offset
            duration = duration_pattern[note_index]
            piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + duration))
            index += 1
    midi.instruments.append(piano)
    midi.write(str(path))
    return str(path)


def review_fill_report(root: Path, *, keep_claim: bool = False) -> dict:
    midi_paths = [
        write_songlike_midi(root / "rank_01.mid", [60, 57, 62, 66, 72, 67, 69, 72]),
        write_songlike_midi(root / "rank_02.mid", [71, 67, 66, 71, 69, 75, 79, 74]),
        write_songlike_midi(root / "rank_03.mid", [67, 66, 69, 71, 67, 62, 63, 67]),
    ]
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_user_listening_review_fill",
        "reviewed_candidates": [
            {
                "rank": index,
                "midi_path": path,
                "wav_path": f"rank_{index:02d}.wav",
            }
            for index, path in enumerate(midi_paths, start=1)
        ],
        "user_listening_review": {
            "status": "reviewed",
            "preferred_rank": 3,
            "overall_decision": "reject_all",
            "primary_failure": "songlike_melody_not_soloing",
        },
        "claim_boundary": {
            "human_audio_preference_claimed": keep_claim,
            "model_direct_candidate_keep_claimed": keep_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloModelDirectSonglikeRejectionAnalysisTest(unittest.TestCase):
    def test_analyze_candidate_detects_repeated_songlike_rhythm_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = write_songlike_midi(Path(temp_dir) / "candidate.mid", [60, 62, 64, 65, 67, 69, 71, 72])
            analysis = analyze_midi_candidate(midi_path, rank=1)

        self.assertEqual(analysis["note_count"], 32)
        self.assertEqual(analysis["bar_count"], 8)
        self.assertEqual(analysis["most_common_note_count_per_bar"], 4)
        self.assertTrue(analysis["four_bar_rhythm_cycle_repeated"])
        self.assertIn("uniform_bar_density", analysis["analysis_flags"])
        self.assertIn("four_notes_per_bar_template", analysis["analysis_flags"])
        self.assertIn("duration_template_monotony", analysis["analysis_flags"])

    def test_builds_rejection_analysis_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_songlike_rejection_analysis_report(
                review_fill_report(root),
                output_dir=root / "out",
            )
            summary = validate_songlike_rejection_analysis_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_analysis_completed=True,
                require_rejection_signals=True,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["uniform_bar_density_count"], 3)
        self.assertEqual(summary["four_notes_per_bar_template_count"], 3)
        self.assertGreaterEqual(summary["duration_template_monotony_count"], 3)
        self.assertEqual(summary["shared_rhythm_signature_count"], 3)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_source_report_with_keep_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBMidiToSoloModelDirectSonglikeRejectionAnalysisError):
                build_songlike_rejection_analysis_report(
                    review_fill_report(root, keep_claim=True),
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
