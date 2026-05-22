from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.run_stage_b_data_motif_generation_compare import (
    build_review_export,
    data_motif_tokens,
    duration_tokens_from_steps,
    fit_duration_tokens_to_positions,
    nearest_allowed_pitch_token,
    normalize_position_deltas,
    parse_baseline_modes,
    straight_grid_tokens,
)
from scripts.run_stage_b_generation_probe import (
    analyze_stage_b_note_grammar,
    build_stage_b_primer,
    extract_stage_b_note_groups,
)
from scripts.stage_b_tokens import (
    TOKEN_NOTE_PITCH_END,
    TOKEN_NOTE_PITCH_START,
    duration_steps_from_token,
    pitch_from_token,
)


def template_report() -> dict:
    return {
        "summary": {
            "top_rhythm_templates": [
                {
                    "count": 3,
                    "key": {
                        "position_deltas": [0, 2, 5, 7],
                        "duration_steps": [2, 1, 3, 2],
                    },
                },
                {
                    "count": 1,
                    "key": {
                        "position_deltas": [0, 3, 4, 9],
                        "duration_steps": [1, 4, 2, 1],
                    },
                },
            ],
            "top_contour_templates": [
                {
                    "count": 2,
                    "key": {
                        "pitch_intervals": [0, 2, 5, 3],
                    },
                },
                {
                    "count": 1,
                    "key": {
                        "pitch_intervals": [0, -2, 1, 0],
                    },
                },
            ],
        }
    }


class StageBDataMotifGenerationCompareTest(unittest.TestCase):
    def test_parse_baseline_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_baseline_modes("hand_written_swing,random")

    def test_normalize_position_deltas_scales_into_slot(self) -> None:
        positions = normalize_position_deltas([0, 3, 9, 12], slot_start=8, slot_size=8)

        self.assertEqual(positions[0], 8)
        self.assertLessEqual(positions[-1], 15)
        self.assertEqual(positions, sorted(set(positions)))

    def test_duration_tokens_from_steps_clamps_count(self) -> None:
        tokens = duration_tokens_from_steps([0, 2, 99], target_count=4)
        steps = [duration_steps_from_token(token) for token in tokens]

        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0], 1)
        self.assertGreaterEqual(steps[-1], 1)

    def test_fit_duration_tokens_to_positions_prevents_overlap(self) -> None:
        tokens = fit_duration_tokens_to_positions([0, 1, 3, 7], [8, 8, 8, 8])
        steps = [duration_steps_from_token(token) for token in tokens]

        self.assertEqual(steps, [1, 2, 4, 4])

    def test_nearest_allowed_pitch_token_avoids_recent_pitch_when_possible(self) -> None:
        allowed = list(range(TOKEN_NOTE_PITCH_START, TOKEN_NOTE_PITCH_END + 1))
        recent_pitch = pitch_from_token(allowed[0])

        token = nearest_allowed_pitch_token(recent_pitch, allowed[:3], recent_pitches=[recent_pitch])

        self.assertNotEqual(pitch_from_token(token), recent_pitch)

    def test_data_motif_tokens_builds_strictly_increasing_solo_line_groups(self) -> None:
        primer = build_stage_b_primer(["Cm7", "Fm7"], 124)
        tokens = data_motif_tokens(
            primer_tokens=primer,
            chords=["Cm7", "Fm7"],
            bars=2,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        positions_by_bar: dict[int, list[int]] = {}
        for group in groups:
            positions_by_bar.setdefault(int(group["bar"]), []).append(int(group["position"]))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        for positions in positions_by_bar.values():
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len(positions), len(set(positions)))

    def test_straight_grid_tokens_stay_on_even_subdivision_grid(self) -> None:
        primer = build_stage_b_primer(["Cm7", "Fm7"], 124)
        tokens = straight_grid_tokens(
            primer_tokens=primer,
            chords=["Cm7", "Fm7"],
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        self.assertEqual({group["position"] % 2 for group in groups}, {0})

    def test_build_review_export_copies_named_mode_files(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_midi = tmp_path / "source.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=124)
            instrument = pretty_midi.Instrument(program=0, name="Solo")
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
            midi.instruments.append(instrument)
            midi.write(str(source_midi))
            samples = {
                "data_motif": [
                    {
                        "sample_index": 1,
                        "sample_seed": 17,
                        "valid": True,
                        "strict_valid": True,
                        "midi_path": str(source_midi),
                        "metrics": {
                            "note_count": 8,
                            "unique_pitch_count": 4,
                            "dead_air_ratio": 0.1,
                        },
                        "rhythm_profile": {
                            "syncopated_onset_ratio": 0.6,
                            "unique_bar_position_pattern_ratio": 1.0,
                            "duration_diversity_ratio": 0.2,
                            "most_common_duration_ratio": 0.4,
                            "ioi_diversity_ratio": 0.2,
                            "most_common_ioi_ratio": 0.4,
                        },
                        "pitch_roles": {
                            "tension_ratio": 0.2,
                            "root_tone_ratio": 0.0,
                        },
                    }
                ]
            }

            manifest = build_review_export(
                samples,
                output_dir=tmp_path / "review",
                top_n=1,
                copy_midi=True,
                chords=["Cm7", "Fm7"],
                bpm=124,
                bars=2,
            )

            copied = Path(manifest["candidates"][0]["review_midi_path"])
            context = Path(manifest["candidates"][0]["context_midi_path"])
            self.assertTrue(copied.exists())
            self.assertTrue(context.exists())
            self.assertIn("data_motif", copied.name)
            self.assertTrue(Path(manifest["chord_guide_midi_path"]).exists())
            self.assertTrue((tmp_path / "review" / "review_manifest.json").exists())
            self.assertTrue((tmp_path / "review" / "review_candidates.md").exists())


if __name__ == "__main__":
    unittest.main()
