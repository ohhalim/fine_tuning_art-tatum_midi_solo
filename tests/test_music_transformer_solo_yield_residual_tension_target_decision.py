from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.assess_stage_b_generic_base_readiness import write_json
from scripts.decide_music_transformer_solo_yield_residual_tension_target import (
    SoloYieldResidualTensionTargetDecisionError,
    build_decision_report,
    select_next_target,
    validate_decision_report,
)


def sample(
    sample_index: int,
    midi_path: Path,
    *,
    dead_air: float,
    direction: float,
    syncopation: float,
    tension: float,
) -> dict:
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
            "syncopated_onset_ratio": syncopation,
        },
        "pitch_roles": {
            "chord_tone_ratio": 0.44,
            "tension_ratio": tension,
        },
    }


def source_package(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_tension_color_balance_repair_package_v1",
        "output_dir": "outputs/source_package",
        "selected_candidates": [
            {
                "dead_air_ratio": 0.60,
                "direction_change_ratio": 0.56,
                "syncopated_onset_ratio": 0.80,
                "tension_ratio": 0.16,
                "note_count": 30,
            },
            {
                "dead_air_ratio": 0.62,
                "direction_change_ratio": 0.58,
                "syncopated_onset_ratio": 0.66,
                "tension_ratio": 0.25,
                "note_count": 31,
            },
        ],
        "readiness": {
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def fixture_sweep(temp_dir: Path, *, tension_safe_count: int) -> dict:
    sample_dir = temp_dir / "samples"
    sample_dir.mkdir()
    midi_paths = []
    for index in range(1, 4):
        midi_path = sample_dir / f"sample_{index}.mid"
        midi_path.write_bytes(f"midi-{index}".encode("ascii"))
        midi_paths.append(midi_path)

    safe_samples = [
        sample(1, midi_paths[0], dead_air=0.60, direction=0.56, syncopation=0.80, tension=0.25),
        sample(2, midi_paths[1], dead_air=0.62, direction=0.58, syncopation=0.80, tension=0.24),
    ][:tension_safe_count]
    unsafe_samples = [
        sample(3, midi_paths[2], dead_air=0.61, direction=0.57, syncopation=0.80, tension=0.16)
    ]
    probe_path = temp_dir / "probe.json"
    write_json(probe_path, {"samples": safe_samples + unsafe_samples})
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


class MusicTransformerSoloYieldResidualTensionTargetDecisionTest(unittest.TestCase):
    def test_selects_syncopation_when_tension_repeat_is_not_feasible(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            report = build_decision_report(
                sweep_report=fixture_sweep(temp_dir, tension_safe_count=1),
                source_package=source_package(),
                output_dir=temp_dir / "decision",
                selected_per_case=2,
                max_dead_air_ratio=0.68,
                min_direction_change_ratio=0.50,
                min_tension_ratio=0.20,
                min_syncopation_ratio=0.70,
                dead_air_watch_max=0.66,
            )
        summary = validate_decision_report(
            report,
            expected_target="rhythm_syncopation_balance_repair",
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["residual_label_counts"]["low_tension_color"], 1)
        self.assertEqual(summary["residual_label_counts"]["low_syncopation"], 1)
        self.assertFalse(summary["tension_repeat_feasible"])
        self.assertEqual(summary["tension_repeat_blocked_cases"], ["minor_backdoor"])
        self.assertEqual(summary["selected_repair_target"], "rhythm_syncopation_balance_repair")

    def test_selects_tension_when_each_case_has_enough_safe_candidates(self) -> None:
        target = select_next_target(
            {"low_tension_color": 1, "low_syncopation": 1},
            [{"case_label": "minor_backdoor", "tension_safe_feasible": True}],
        )

        self.assertEqual(target, "tension_color_balance_repair")

    def test_validation_rejects_unexpected_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            report = build_decision_report(
                sweep_report=fixture_sweep(temp_dir, tension_safe_count=1),
                source_package=source_package(),
                output_dir=temp_dir / "decision",
                selected_per_case=2,
                max_dead_air_ratio=0.68,
                min_direction_change_ratio=0.50,
                min_tension_ratio=0.20,
                min_syncopation_ratio=0.70,
                dead_air_watch_max=0.66,
            )

        with self.assertRaises(SoloYieldResidualTensionTargetDecisionError):
            validate_decision_report(
                report,
                expected_target="tension_color_balance_repair",
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp_dir = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualTensionTargetDecisionError):
                build_decision_report(
                    sweep_report=fixture_sweep(temp_dir, tension_safe_count=1),
                    source_package=source_package(quality_claim=True),
                    output_dir=temp_dir / "decision",
                    selected_per_case=2,
                    max_dead_air_ratio=0.68,
                    min_direction_change_ratio=0.50,
                    min_tension_ratio=0.20,
                    min_syncopation_ratio=0.70,
                    dead_air_watch_max=0.66,
                )


if __name__ == "__main__":
    unittest.main()
