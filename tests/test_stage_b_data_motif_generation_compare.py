from __future__ import annotations

import unittest

from scripts.run_stage_b_data_motif_generation_compare import (
    data_motif_tokens,
    duration_tokens_from_steps,
    fit_duration_tokens_to_positions,
    nearest_allowed_pitch_token,
    normalize_position_deltas,
    parse_baseline_modes,
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


if __name__ == "__main__":
    unittest.main()
