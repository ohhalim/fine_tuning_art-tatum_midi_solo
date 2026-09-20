from __future__ import annotations

import unittest

from scripts.generate import (
    truncate_tokens_preserving_velocity,
    VELOCITY_TOKEN_END,
    VELOCITY_TOKEN_START,
    _carry_velocity_into_block,
)
from midi_processor.processor import RANGE_NOTE_ON, decode_midi


class VelocityCarryTests(unittest.TestCase):
    def test_block_without_velocity_token_decodes_to_unplayable_zero(self) -> None:
        """This is the defect the carry exists to prevent."""
        block = [60, RANGE_NOTE_ON * 2 + 9, RANGE_NOTE_ON + 60]
        notes = [n for i in decode_midi(list(block)).instruments for n in i.notes]

        self.assertTrue(notes)
        self.assertTrue(all(n.velocity == 0 for n in notes))

    def test_carry_restores_primer_velocity_when_block_has_none(self) -> None:
        primer = [VELOCITY_TOKEN_START + 10, 60, VELOCITY_TOKEN_START + 11]
        block = [60, RANGE_NOTE_ON * 2 + 9, RANGE_NOTE_ON + 60]

        carried = _carry_velocity_into_block(primer, block)

        self.assertTrue(carried)
        self.assertEqual(VELOCITY_TOKEN_START + 11, block[0])
        notes = [n for i in decode_midi(list(block)).instruments for n in i.notes]
        self.assertTrue(all(1 <= n.velocity <= 127 for n in notes))

    def test_carry_is_skipped_when_block_sets_velocity_before_first_note(self) -> None:
        primer = [VELOCITY_TOKEN_START + 11]
        block = [VELOCITY_TOKEN_START + 15, 60, RANGE_NOTE_ON + 60]
        original = list(block)

        self.assertFalse(_carry_velocity_into_block(primer, block))
        self.assertEqual(original, block)

    def test_carry_applies_when_velocity_token_lands_after_a_note(self) -> None:
        """A late velocity token still leaves the notes before it at velocity 0."""
        primer = [VELOCITY_TOKEN_START + 11]
        shift = RANGE_NOTE_ON * 2 + 9  # 10 time steps
        block = [60, shift, RANGE_NOTE_ON + 60,
                 VELOCITY_TOKEN_START + 12, 62, shift, RANGE_NOTE_ON + 62]

        self.assertTrue(_carry_velocity_into_block(primer, block))
        self.assertEqual(VELOCITY_TOKEN_START + 11, block[0])
        notes = [n for i in decode_midi(list(block)).instruments for n in i.notes]
        self.assertTrue(notes)
        self.assertTrue(all(1 <= n.velocity <= 127 for n in notes))

    def test_carry_is_skipped_when_block_has_no_note_on(self) -> None:
        block = [RANGE_NOTE_ON * 2 + 9]
        original = list(block)

        self.assertFalse(_carry_velocity_into_block([VELOCITY_TOKEN_START + 11], block))
        self.assertEqual(original, block)

    def test_carry_is_skipped_when_primer_has_no_velocity(self) -> None:
        block = [60, RANGE_NOTE_ON + 60]
        original = list(block)

        self.assertFalse(_carry_velocity_into_block([60, 61], block))
        self.assertEqual(original, block)

    def test_velocity_token_range_matches_processor_layout(self) -> None:
        self.assertEqual(356, VELOCITY_TOKEN_START)
        self.assertEqual(388, VELOCITY_TOKEN_END)


if __name__ == "__main__":
    unittest.main()


class TruncationVelocityTests(unittest.TestCase):
    """Tail-truncating Stage A tokens must not drop carried velocity."""

    def test_short_sequence_is_returned_unchanged(self) -> None:
        tokens = [VELOCITY_TOKEN_START + 12, 60, RANGE_NOTE_ON + 60]
        self.assertEqual(tokens, truncate_tokens_preserving_velocity(tokens, 10))

    def test_lone_leading_velocity_token_survives_truncation(self) -> None:
        """Steady playing emits one velocity token, at the very start."""
        tokens = [VELOCITY_TOKEN_START + 12] + [60, RANGE_NOTE_ON + 60] * 20

        result = truncate_tokens_preserving_velocity(tokens, 8)

        self.assertEqual(8, len(result))
        self.assertEqual(VELOCITY_TOKEN_START + 12, result[0])

    def test_velocity_already_ahead_of_the_first_note_is_left_alone(self) -> None:
        tokens = [60, RANGE_NOTE_ON + 60] * 5 + [VELOCITY_TOKEN_START + 9, 62, RANGE_NOTE_ON + 62]

        result = truncate_tokens_preserving_velocity(tokens, 3)

        self.assertEqual([VELOCITY_TOKEN_START + 9, 62, RANGE_NOTE_ON + 62], result)

    def test_no_velocity_anywhere_returns_the_plain_tail(self) -> None:
        tokens = [60, RANGE_NOTE_ON + 60] * 10
        self.assertEqual([60, RANGE_NOTE_ON + 60], truncate_tokens_preserving_velocity(tokens, 2))

    def test_zero_budget_returns_nothing(self) -> None:
        self.assertEqual([], truncate_tokens_preserving_velocity([60, 61], 0))

    def test_truncated_tokens_decode_without_silent_notes(self) -> None:
        tokens = [VELOCITY_TOKEN_START + 12]
        for pitch in range(60, 72):
            tokens += [pitch, RANGE_NOTE_ON * 2 + 9, RANGE_NOTE_ON + pitch]

        result = truncate_tokens_preserving_velocity(tokens, 12)
        notes = [n for i in decode_midi(list(result)).instruments for n in i.notes]

        self.assertTrue(notes)
        self.assertTrue(all(n.velocity > 0 for n in notes), [n.velocity for n in notes])
