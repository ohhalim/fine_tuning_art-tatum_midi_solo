from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.decide_music_transformer_solo_yield_density_aftercare_objective_next import (
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import midi_profile
from scripts.run_music_transformer_solo_yield_density_aftercare_sweep import (
    SCHEMA_VERSION as SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
)
from scripts.run_music_transformer_solo_yield_interval_contour_aftercare_sweep import (
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    SoloYieldIntervalContourAftercareSweepError,
    build_repair_sweep,
    validate_report,
)


CHORDS = "Cm7,F7,Bbmaj7,Ebmaj7"


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.12)
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


def source_repair_sweep(rows: list[dict], *, quality_claim: bool = False) -> dict:
    return {
        "schema_version": SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
        "output_dir": "outputs/source",
        "candidate_repairs": rows,
        "aggregate": {
            "candidate_count": len(rows),
            "target_supported": True,
            "low_note_count_after": 0,
            "chord_tone_ratio_decrease_count": 0,
            "weak_direction_change_count_after": 0,
            "final_landing_not_chord_tone_count_after": 0,
            "wide_interval_review_count_after": 1,
        },
        "readiness": {
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def objective_decision(*, selected_next_target: str = "interval_contour_aftercare") -> dict:
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


class MusicTransformerSoloYieldIntervalContourAftercareSweepTest(unittest.TestCase):
    def test_repairs_wide_interval_without_guard_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [
                source_row(root, 1, [70, 70, 70, 62, 64, 68, 70, 70]),
                source_row(root, 2, [63, 67, 70, 67, 70, 72, 70, 70]),
            ]
            report = build_repair_sweep(
                source_repair_sweep=source_repair_sweep(rows),
                objective_decision=objective_decision(),
                output_dir=root / "repair",
                min_note_count=8,
                min_direction_change=0.0,
            )
        summary = validate_report(report, min_candidates=2)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["wide_interval_review_count_before"], 1)
        self.assertEqual(summary["wide_interval_review_count_after"], 0)
        self.assertGreater(summary["adjusted_note_total"], 0)
        self.assertLessEqual(summary["max_pitch_shift"], 2)
        self.assertEqual(summary["chord_tone_ratio_decrease_count"], 0)
        self.assertEqual(summary["low_note_count_after"], 0)
        self.assertEqual(summary["final_landing_not_chord_tone_count_after"], 0)
        self.assertTrue(summary["target_supported"])
        self.assertEqual(summary["selected_next_target"], SELECTED_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [source_row(root, 1, [70, 70, 70, 62, 64, 68, 70, 70])]
            with self.assertRaises(SoloYieldIntervalContourAftercareSweepError):
                build_repair_sweep(
                    source_repair_sweep=source_repair_sweep(rows, quality_claim=True),
                    objective_decision=objective_decision(),
                    output_dir=root / "repair",
                )

    def test_rejects_wrong_objective_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [source_row(root, 1, [70, 70, 70, 62, 64, 68, 70, 70])]
            with self.assertRaises(SoloYieldIntervalContourAftercareSweepError):
                build_repair_sweep(
                    source_repair_sweep=source_repair_sweep(rows),
                    objective_decision=objective_decision(selected_next_target="density_aftercare"),
                    output_dir=root / "repair",
                )


if __name__ == "__main__":
    unittest.main()
