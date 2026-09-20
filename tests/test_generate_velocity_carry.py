from __future__ import annotations

import unittest

from scripts.generate import (
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
