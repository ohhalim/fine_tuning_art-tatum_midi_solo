from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankCliUserInputSmokeError,
    build_user_input_smoke_report,
    validate_user_input_smoke_report,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (
    BOUNDARY as CLI_PACKAGE_BOUNDARY,
    NEXT_BOUNDARY as CLI_PACKAGE_NEXT_BOUNDARY,
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


def package_report(root: Path, *, input_path: Path) -> dict:
    candidates = []
    for rank in range(1, 4):
        midi_path = write_test_midi(root / f"rank_{rank}.mid")
        candidates.append(
            {
                "rank": rank,
                "sample_seed": 630 + rank,
                "repaired_midi_path": str(midi_path),
                "objective_supported": True,
                "note_count": 96,
                "unique_pitch_count": 20,
                "max_simultaneous_notes": 1,
                "dead_air_ratio": 0.2 + rank * 0.01,
                "phrase_coverage_ratio": 1.0,
            }
        )
    return {
        "boundary": CLI_PACKAGE_BOUNDARY,
        "input": {"midi_path": str(input_path)},
        "objective_summary": {
            "candidate_count": 3,
            "objective_supported_candidate_count": 3,
            "all_candidates_objective_supported": True,
            "min_dead_air_ratio": 0.21,
            "max_dead_air_ratio": 0.23,
            "input_context_bars": 12,
            "phrase_bank_exported_candidate_count": 3,
        },
        "candidate_manifest": candidates,
        "readiness": {
            "cli_mvp_package_completed": True,
            "ranked_repaired_midi_exported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": CLI_PACKAGE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseBankCliUserInputSmokeTest(unittest.TestCase):
    def test_builds_user_input_smoke_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = write_test_midi(root / "external_input.mid")
            report = build_user_input_smoke_report(
                cli_package_report=package_report(root, input_path=input_path),
                package_report_path=root / "package.json",
                output_dir=root / "smoke",
                issue_number=654,
                expected_input_midi=input_path,
                min_candidate_count=3,
                require_explicit_input=True,
            )
            summary = validate_user_input_smoke_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_candidate_count=3,
                require_explicit_input=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["candidate_count"], 3)
            self.assertTrue(summary["explicit_input_used"])
            self.assertEqual(summary["repaired_midi_file_count"], 3)
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_auto_fixture_when_explicit_input_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_path = write_test_midi(root / "input" / "fixture.mid")

            with self.assertRaises(StageBMidiToSoloPhraseBankCliUserInputSmokeError):
                build_user_input_smoke_report(
                    cli_package_report=package_report(root, input_path=fixture_path),
                    package_report_path=root / "package.json",
                    output_dir=root / "smoke",
                    issue_number=654,
                    expected_input_midi=fixture_path,
                    min_candidate_count=3,
                    require_explicit_input=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke")


if __name__ == "__main__":
    unittest.main()
