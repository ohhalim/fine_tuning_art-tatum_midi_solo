from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.diagnose_stage_b_midi_to_solo_model_direct_phrase_quality import (
    BOUNDARY,
    PITCH_REPAIR_BOUNDARY,
    StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError,
    build_phrase_quality_diagnostics_report,
    note_metrics_for_path,
    validate_phrase_quality_diagnostics_report,
)


def write_midi(path: Path, pitches: list[int], *, step: float = 0.5, duration: float = 0.25) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    for index, pitch in enumerate(pitches):
        start = index * step
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=start + duration))
    midi.instruments.append(piano)
    midi.write(str(path))
    return str(path)


def evidence_report(root: Path, *, quality_claim: bool = False) -> dict:
    midi_paths = [
        write_midi(root / "sample_1.mid", [60, 60, 72, 61, 73, 62, 74, 63]),
        write_midi(root / "sample_2.mid", [64, 76, 65, 77, 66, 78, 67, 79]),
        write_midi(root / "sample_3.mid", [67, 67, 79, 68, 80, 69, 81, 70]),
    ]
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation",
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation",
            "model_direct_midi_to_wav_technical_path_completed": True,
            "model_direct_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
        },
        "objective_evidence": {
            "midi_paths": midi_paths,
        },
    }


class StageBMidiToSoloModelDirectPhraseQualityDiagnosticsTest(unittest.TestCase):
    def test_note_metrics_detect_pitch_and_rhythm_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = write_midi(Path(temp_dir) / "candidate.mid", [60, 60, 72, 61, 73, 62, 74, 63])
            metrics = note_metrics_for_path(midi_path, dead_air_threshold_seconds=0.4)

        self.assertEqual(metrics["note_count"], 8)
        self.assertEqual(metrics["adjacent_pitch_repeats"], 1)
        self.assertGreaterEqual(metrics["max_interval"], 12)
        self.assertIn("adjacent_pitch_repetition", metrics["diagnostic_flags"])
        self.assertIn("wide_interval_contour", metrics["diagnostic_flags"])
        self.assertIn("duration_monotony", metrics["diagnostic_flags"])

    def test_builds_diagnostics_and_routes_pitch_repair_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_phrase_quality_diagnostics_report(
                evidence_report=evidence_report(root),
                output_dir=root / "out",
                issue_number=505,
                dead_air_threshold_seconds=0.4,
            )
            summary = validate_phrase_quality_diagnostics_report(
                report,
                expected_boundary=BOUNDARY,
                require_diagnostics_completed=True,
                require_no_quality_claim=True,
                min_candidate_count=3,
            )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["next_boundary"], PITCH_REPAIR_BOUNDARY)
        self.assertGreaterEqual(summary["flag_counts"]["wide_interval_contour"], 1)
        self.assertGreaterEqual(summary["adjacent_pitch_repeat_total"], 1)
        self.assertFalse(summary["model_direct_generation_quality_claimed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(StageBMidiToSoloModelDirectPhraseQualityDiagnosticsError):
                build_phrase_quality_diagnostics_report(
                    evidence_report=evidence_report(root, quality_claim=True),
                    output_dir=root / "out",
                    issue_number=505,
                    dead_air_threshold_seconds=0.4,
                )


if __name__ == "__main__":
    unittest.main()
