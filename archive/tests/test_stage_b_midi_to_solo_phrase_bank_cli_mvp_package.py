from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankCliMvpPackageError,
    build_cli_mvp_package_report,
    source_candidates_from_phrase_bank_report,
    validate_cli_mvp_package_report,
)


def write_test_midi(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    for index in range(4):
        start = index * 0.25
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=60 + index, start=start, end=start + 0.18)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))
    return path


def repaired_candidate(rank: int, path: Path) -> dict:
    return {
        "rank": rank,
        "sample_seed": 630 + rank,
        "source_midi_path": str(path),
        "repaired_midi_path": str(path),
        "repaired_metrics": {
            "note_count": 96,
            "unique_pitch_count": 20,
            "max_simultaneous_notes": 1,
            "dead_air_ratio": 0.2 + rank * 0.01,
            "phrase_coverage_ratio": 1.0,
        },
        "dead_air_gain": 0.35,
        "note_count_gain": 32,
        "repair_gate": {
            "qualified": True,
            "flags": [],
        },
    }


class StageBMidiToSoloPhraseBankCliMvpPackageTest(unittest.TestCase):
    def test_builds_cli_package_manifest_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_midi = write_test_midi(root / "input.mid")
            candidate_paths = [write_test_midi(root / f"rank_{rank}.mid") for rank in range(1, 4)]
            report = build_cli_mvp_package_report(
                input_midi=input_midi,
                output_dir=root / "package",
                issue_number=652,
                context_summary={"context_bars": 8},
                phrase_bank_summary={
                    "exported_candidate_count": 3,
                    "exported_qualified_candidate_count": 3,
                },
                repaired_candidates=[
                    repaired_candidate(rank, candidate_paths[rank - 1]) for rank in range(1, 4)
                ],
                context_report_path=root / "context.json",
                phrase_bank_report_path=root / "phrase_bank.json",
                cli_command="python script.py --input_midi input.mid",
            )
            summary = validate_cli_mvp_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_candidate_count=3,
                require_cli_ready=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["candidate_count"], 3)
            self.assertEqual(summary["objective_supported_candidate_count"], 3)
            self.assertTrue(summary["cli_mvp_package_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path = write_test_midi(root / "rank_1.mid")
            report = build_cli_mvp_package_report(
                input_midi=write_test_midi(root / "input.mid"),
                output_dir=root / "package",
                issue_number=652,
                context_summary={"context_bars": 8},
                phrase_bank_summary={
                    "exported_candidate_count": 1,
                    "exported_qualified_candidate_count": 1,
                },
                repaired_candidates=[repaired_candidate(1, candidate_path)],
                context_report_path=root / "context.json",
                phrase_bank_report_path=root / "phrase_bank.json",
                cli_command="python script.py",
            )
            report["readiness"]["midi_to_solo_musical_quality_claimed"] = True

            with self.assertRaises(StageBMidiToSoloPhraseBankCliMvpPackageError):
                validate_cli_mvp_package_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    min_candidate_count=1,
                    require_cli_ready=True,
                    require_no_quality_claim=True,
                )

    def test_source_candidate_contract_requires_existing_exported_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            midi_path = write_test_midi(root / "export.mid")
            report = {
                "top_candidates": [
                    {
                        "rank": 1,
                        "sample_seed": 632,
                        "export_midi_path": str(midi_path),
                        "exported_metrics": {"note_count": 24},
                    }
                ]
            }

            rows = source_candidates_from_phrase_bank_report(report, min_candidate_count=1)

            self.assertEqual(rows[0]["rank"], 1)
            self.assertEqual(rows[0]["sample_seed"], 632)
            self.assertEqual(rows[0]["midi_path"], str(midi_path))

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_mvp_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke")


if __name__ == "__main__":
    unittest.main()
