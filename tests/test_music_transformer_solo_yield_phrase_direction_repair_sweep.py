from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_music_transformer_solo_yield_chord_tone_landing_objective_next import (
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import midi_profile
from scripts.run_music_transformer_solo_yield_chord_tone_landing_repair_sweep import (
    SCHEMA_VERSION as SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
)
from scripts.run_music_transformer_solo_yield_phrase_direction_repair_sweep import (
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldPhraseDirectionRepairSweepError,
    build_repair_sweep,
    validate_report,
)


CHORDS = "Cm7,Fm7,Bb7,Ebmaj7"


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.24
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.18)
        )
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def source_row(root: Path, index: int, pitches: list[int]) -> dict:
    midi_path = root / f"candidate_{index}.mid"
    write_midi(midi_path, pitches)
    profile = midi_profile({"review_midi_path": str(midi_path), "chords": CHORDS})
    return {
        "review_index": index,
        "case_label": f"case_{index}",
        "chords": CHORDS,
        "repaired_midi_path": str(midi_path),
        "source_wav_path": str(root / f"candidate_{index}.wav"),
        "after_profile": profile,
    }


def source_repair_sweep(rows: list[dict]) -> dict:
    return {
        "schema_version": SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
        "output_dir": "outputs/source",
        "candidate_repairs": rows,
        "aggregate": {
            "candidate_count": len(rows),
            "target_supported": True,
            "final_landing_not_chord_tone_count_after": 0,
        },
        "readiness": {
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def objective_decision(*, selected_next_target: str = "phrase_direction_repair") -> dict:
    return {
        "schema_version": OBJECTIVE_DECISION_SCHEMA_VERSION,
        "output_dir": "outputs/objective",
        "readiness": {
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "selected_next_target": selected_next_target,
        },
    }


class MusicTransformerSoloYieldPhraseDirectionRepairSweepTest(unittest.TestCase):
    def test_repairs_weak_direction_without_chord_tone_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [
                source_row(root, 1, [60, 62, 64, 65, 67, 69, 70, 63]),
                source_row(root, 2, [60, 61, 63, 65, 66, 68, 70, 63]),
                source_row(root, 3, [60, 64, 61, 65, 62, 67, 63, 67]),
            ]
            report = build_repair_sweep(
                source_repair_sweep=source_repair_sweep(rows),
                objective_decision=objective_decision(),
                output_dir=root / "repair",
                min_direction_change=0.50,
            )
        summary = validate_report(report, min_candidates=3)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertGreater(summary["weak_direction_change_count_before"], 0)
        self.assertEqual(summary["weak_direction_change_count_after"], 0)
        self.assertEqual(summary["chord_tone_ratio_decrease_count"], 0)
        self.assertEqual(summary["final_landing_not_chord_tone_count_after"], 0)
        self.assertTrue(summary["target_supported"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_wrong_objective_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [source_row(root, 1, [60, 62, 64, 65, 67, 69, 70, 63])]
            with self.assertRaises(SoloYieldPhraseDirectionRepairSweepError):
                build_repair_sweep(
                    source_repair_sweep=source_repair_sweep(rows),
                    objective_decision=objective_decision(selected_next_target="chord_role_balance_repair"),
                    output_dir=root / "repair",
                )


if __name__ == "__main__":
    unittest.main()
