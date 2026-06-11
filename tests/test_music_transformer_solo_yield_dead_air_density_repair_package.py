from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.assess_stage_b_generic_base_readiness import write_json
from scripts.build_music_transformer_solo_yield_dead_air_density_repair_package import (
    SoloYieldDeadAirDensityRepairPackageError,
    build_repair_package,
    select_repair_candidates,
    validate_repair_package,
)


def sample(sample_index: int, midi_path: Path, *, dead_air: float, direction: float = 0.55) -> dict:
    return {
        "sample_index": sample_index,
        "sample_seed": 1000 + sample_index,
        "midi_path": str(midi_path),
        "strict_valid": True,
        "valid": True,
        "grammar_gate_passed": True,
        "metrics": {
            "note_count": 30 + sample_index,
            "unique_pitch_count": 12,
            "dead_air_ratio": dead_air,
        },
        "phrase_contour": {
            "direction_change_ratio": direction,
        },
        "rhythm_profile": {
            "syncopated_onset_ratio": 0.80,
        },
        "pitch_roles": {
            "chord_tone_ratio": 0.44,
            "tension_ratio": 0.24,
        },
    }


def source_listening_package(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_package_v1",
        "output_dir": "outputs/source_listening",
        "candidate_count": 2,
        "candidates": [
            {
                "dead_air_ratio": 0.72,
                "direction_change_ratio": 0.60,
                "tension_ratio": 0.24,
                "note_count": 30,
            },
            {
                "dead_air_ratio": 0.70,
                "direction_change_ratio": 0.55,
                "tension_ratio": 0.24,
                "note_count": 31,
            },
        ],
        "readiness": {
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def fixture_sweep(temp_dir: Path) -> dict:
    sample_dir = temp_dir / "samples"
    sample_dir.mkdir()
    midi_paths = []
    for index in range(1, 5):
        midi_path = sample_dir / f"sample_{index}.mid"
        midi_path.write_bytes(f"midi-{index}".encode("ascii"))
        midi_paths.append(midi_path)

    probe_report = {
        "samples": [
            sample(1, midi_paths[0], dead_air=0.74, direction=0.70),
            sample(2, midi_paths[1], dead_air=0.54, direction=0.45),
            sample(3, midi_paths[2], dead_air=0.58, direction=0.60),
            sample(4, midi_paths[3], dead_air=0.62, direction=0.55),
        ]
    }
    probe_path = temp_dir / "probe.json"
    write_json(probe_path, probe_report)
    return {
        "schema_version": "music_transformer_solo_yield_sweep_v1",
        "output_dir": "outputs/sweep",
        "cases": [
            {
                "label": "minor_backdoor",
                "chords": "Cm7,F7,Bbmaj7,Ebmaj7",
                "seed": 2300,
                "probe_report_path": str(probe_path),
            }
        ],
        "readiness": {
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldDeadAirDensityRepairPackageTest(unittest.TestCase):
    def test_selects_low_dead_air_candidates_from_probe_pool(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            selected = select_repair_candidates(fixture_sweep(temp_dir), selected_per_case=2)

        self.assertEqual([item["sample_index"] for item in selected], [2, 3])
        self.assertEqual([round(item["dead_air_ratio"], 2) for item in selected], [0.54, 0.58])

    def test_builds_repair_package_with_dead_air_reduction_without_render(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            report = build_repair_package(
                sweep_report=fixture_sweep(temp_dir),
                source_listening_package=source_listening_package(),
                output_dir=temp_dir / "repair",
                selected_per_case=2,
                major_dead_air_threshold=0.68,
                renderer="",
                soundfont="",
                sample_rate=44100,
                render_audio=False,
            )
        summary = validate_repair_package(
            report,
            min_candidate_count=2,
            require_dead_air_avg_reduced=True,
            require_wav_rendered=False,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["candidate_midi_files_copied"], 2)
        self.assertEqual(summary["candidate_wav_files_rendered"], 0)
        self.assertGreater(summary["dead_air_avg_delta"], 0.0)
        self.assertEqual(summary["source_dead_air_high_count"], 2)
        self.assertEqual(summary["repair_dead_air_high_count"], 0)
        self.assertTrue(summary["dead_air_avg_reduced"])
        self.assertFalse(summary["technical_wav_render_completed"])
        self.assertFalse(summary["musical_quality_claimed"])

    def test_validation_requires_wav_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            report = build_repair_package(
                sweep_report=fixture_sweep(temp_dir),
                source_listening_package=source_listening_package(),
                output_dir=temp_dir / "repair",
                selected_per_case=2,
                major_dead_air_threshold=0.68,
                renderer="",
                soundfont="",
                sample_rate=44100,
                render_audio=False,
            )

        with self.assertRaises(SoloYieldDeadAirDensityRepairPackageError):
            validate_repair_package(
                report,
                min_candidate_count=2,
                require_dead_air_avg_reduced=True,
                require_wav_rendered=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            with self.assertRaises(SoloYieldDeadAirDensityRepairPackageError):
                build_repair_package(
                    sweep_report=fixture_sweep(temp_dir),
                    source_listening_package=source_listening_package(quality_claim=True),
                    output_dir=temp_dir / "repair",
                    selected_per_case=2,
                    major_dead_air_threshold=0.68,
                    renderer="",
                    soundfont="",
                    sample_rate=44100,
                    render_audio=False,
                )


if __name__ == "__main__":
    unittest.main()
