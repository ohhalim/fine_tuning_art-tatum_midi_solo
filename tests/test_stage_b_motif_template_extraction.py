from __future__ import annotations

import unittest

from scripts.run_stage_b_motif_template_extraction import (
    extract_motif_templates_from_tokens,
    motif_from_groups,
    summarize_motif_templates,
)
from scripts.stage_b_tokens import (
    TOKEN_BAR,
    TOKEN_END,
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    position_token,
)


def phrase_tokens(pitch_shift: int = 0) -> list[int]:
    return [
        *chord_tokens("Cmaj7"),
        position_token(0),
        note_velocity_token(4),
        note_pitch_token(60 + pitch_shift),
        note_duration_token(2),
        position_token(3),
        note_velocity_token(4),
        note_pitch_token(62 + pitch_shift),
        note_duration_token(1),
        position_token(7),
        note_velocity_token(4),
        note_pitch_token(65 + pitch_shift),
        note_duration_token(3),
        TOKEN_BAR,
        *chord_tokens("F7"),
        position_token(1),
        note_velocity_token(4),
        note_pitch_token(64 + pitch_shift),
        note_duration_token(2),
        position_token(5),
        note_velocity_token(4),
        note_pitch_token(67 + pitch_shift),
        note_duration_token(1),
        TOKEN_END,
    ]


def same_onset_phrase_tokens() -> list[int]:
    return [
        *chord_tokens("Cmaj7"),
        position_token(0),
        note_velocity_token(4),
        note_pitch_token(60),
        note_duration_token(2),
        position_token(0),
        note_velocity_token(4),
        note_pitch_token(64),
        note_duration_token(2),
        position_token(3),
        note_velocity_token(4),
        note_pitch_token(67),
        note_duration_token(2),
        position_token(5),
        note_velocity_token(4),
        note_pitch_token(69),
        note_duration_token(2),
        TOKEN_END,
    ]


class StageBMotifTemplateExtractionTest(unittest.TestCase):
    def test_extract_motif_templates_from_tokens_slides_note_groups(self) -> None:
        motifs = extract_motif_templates_from_tokens(
            phrase_tokens(),
            source_record="sample.npy",
            motif_length=3,
            max_bar_span=2,
        )

        self.assertEqual(len(motifs), 3)
        self.assertEqual(motifs[0]["position_deltas"], [0, 3, 7])
        self.assertEqual(motifs[0]["duration_steps"], [2, 1, 3])
        self.assertEqual(motifs[0]["pitch_intervals"], [0, 2, 5])

    def test_extract_motif_templates_filters_same_onset_blocks_by_default(self) -> None:
        motifs = extract_motif_templates_from_tokens(
            same_onset_phrase_tokens(),
            source_record="same_onset.npy",
            motif_length=4,
            max_bar_span=1,
        )

        self.assertEqual(motifs, [])

    def test_extract_motif_templates_can_allow_same_onset_for_diagnostics(self) -> None:
        motifs = extract_motif_templates_from_tokens(
            same_onset_phrase_tokens(),
            source_record="same_onset.npy",
            motif_length=4,
            max_bar_span=1,
            require_strictly_increasing_onsets=False,
        )

        self.assertEqual(len(motifs), 1)
        self.assertEqual(motifs[0]["position_deltas"], [0, 0, 3, 5])

    def test_pitch_intervals_are_transposition_invariant(self) -> None:
        base = extract_motif_templates_from_tokens(
            phrase_tokens(0),
            source_record="base.npy",
            motif_length=4,
            max_bar_span=2,
        )[0]
        shifted = extract_motif_templates_from_tokens(
            phrase_tokens(5),
            source_record="shifted.npy",
            motif_length=4,
            max_bar_span=2,
        )[0]

        self.assertEqual(base["pitch_intervals"], shifted["pitch_intervals"])
        self.assertEqual(base["melodic_intervals"], shifted["melodic_intervals"])

    def test_motif_from_groups_reports_syncopation_and_direction(self) -> None:
        motifs = extract_motif_templates_from_tokens(
            phrase_tokens(),
            source_record="sample.npy",
            motif_length=4,
            max_bar_span=2,
        )

        motif = motifs[0]

        self.assertGreater(motif["syncopated_onset_ratio"], 0.5)
        self.assertGreaterEqual(motif["direction_change_ratio"], 0.0)
        self.assertEqual(motif["bar_span"], 2)

    def test_summarize_motif_templates_ranks_repeated_templates(self) -> None:
        motifs = extract_motif_templates_from_tokens(
            phrase_tokens(),
            source_record="a.npy",
            motif_length=3,
            max_bar_span=2,
        )
        motifs.extend(
            extract_motif_templates_from_tokens(
                phrase_tokens(2),
                source_record="b.npy",
                motif_length=3,
                max_bar_span=2,
            )
        )

        summary = summarize_motif_templates(motifs, top_n=2)

        self.assertEqual(summary["source_record_count"], 2)
        self.assertGreater(summary["motif_count"], 0)
        self.assertEqual(summary["top_rhythm_templates"][0]["count"], 2)
        self.assertEqual(summary["top_contour_templates"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
