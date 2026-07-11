from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from inference.app.metrics import compute_midi_metrics
from inference.app.schemas import GenerationRequest
from scripts.decide_stage_b_midi_to_solo_phrase_bank_objective_next import BOUNDARY as OBJECTIVE_NEXT_BOUNDARY
from scripts.run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankDeadAirDensityRepairError,
    build_dead_air_density_repaired_notes,
    build_repair_probe_report,
    per_bar_note_counts,
    validate_repair_probe_report,
    write_repaired_midi,
)


def request() -> GenerationRequest:
    return GenerationRequest(
        bpm=120,
        chord_progression=["Cmaj7", "F7", "G7", "Cmaj7", "Cmaj7", "Cmaj7", "Cmaj7", "Cmaj7"],
        bars=8,
        density="medium",
        energy="mid",
    )


def uniform_source_notes() -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for index in range(64):
        start = index * 0.25
        pitch = 60 + (index % 7)
        notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + 0.12))
    return notes


def write_source_midi(path: Path, notes: list[pretty_midi.Note]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    instrument.notes = notes
    midi.instruments.append(instrument)
    midi.write(str(path))


def source_objective_report(source_path: Path) -> dict:
    req = request()
    source_metrics = compute_midi_metrics(source_path, 0, fallback_used=True, request=req).to_dict()
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "candidate_reviews": [
            {
                "rank": 1,
                "sample_seed": 635,
                "midi_path": str(source_path),
                "objective_metrics": source_metrics,
            }
        ],
        "objective_summary": {
            "preference_fill_allowed": False,
        },
        "readiness": {
            "phrase_bank_repair_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseBankDeadAirDensityRepairTest(unittest.TestCase):
    def test_repair_notes_reduce_dead_air_and_vary_density(self) -> None:
        req = request()
        source_notes = uniform_source_notes()
        repaired_notes, additions = build_dead_air_density_repaired_notes(
            source_notes,
            request=req,
            additions_per_bar=[3, 5, 2, 6, 3, 5, 2, 6],
            dead_air_threshold_sec=0.18,
            min_start_separation_sec=0.04,
        )

        self.assertGreaterEqual(len(additions), 16)
        self.assertGreater(len(repaired_notes), len(source_notes))
        self.assertGreaterEqual(len(set(per_bar_note_counts(repaired_notes, req).values())), 3)

    def test_probe_routes_qualified_repair_to_audio_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.mid"
            write_source_midi(source_path, uniform_source_notes())

            report = build_repair_probe_report(
                objective_next_report=source_objective_report(source_path),
                output_dir=tmp_path / "out",
                issue_number=642,
                bpm=120,
                bars=8,
                additions_per_bar=[3, 5, 2, 6, 3, 5, 2, 6],
                dead_air_threshold_sec=0.18,
                min_start_separation_sec=0.04,
                min_dead_air_gain=0.15,
                max_dead_air_ratio=0.45,
                min_unique_density_patterns=3,
                min_note_count_gain=16,
            )
            summary = validate_repair_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_probe_completed=True,
                require_target_passed=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["repaired_candidate_count"], 1)
            self.assertEqual(summary["qualified_repaired_candidate_count"], 1)
            self.assertTrue(summary["repair_probe_target_passed"])
            self.assertLessEqual(summary["max_repaired_dead_air_ratio"], 0.45)

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.mid"
            write_source_midi(source_path, uniform_source_notes())
            report = source_objective_report(source_path)
            report["readiness"]["human_audio_preference_claimed"] = True

            with self.assertRaises(StageBMidiToSoloPhraseBankDeadAirDensityRepairError):
                build_repair_probe_report(
                    objective_next_report=report,
                    output_dir=tmp_path / "out",
                    issue_number=642,
                    bpm=120,
                    bars=8,
                    additions_per_bar=[3, 5, 2, 6, 3, 5, 2, 6],
                    dead_air_threshold_sec=0.18,
                    min_start_separation_sec=0.04,
                    min_dead_air_gain=0.15,
                    max_dead_air_ratio=0.45,
                    min_unique_density_patterns=3,
                    min_note_count_gain=16,
                )

    def test_write_repaired_midi_outputs_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidate.mid"
            write_repaired_midi(uniform_source_notes()[:4], out, bpm=120)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
