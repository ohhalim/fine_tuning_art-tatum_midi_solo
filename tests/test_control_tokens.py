from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pretty_midi

from scripts.control_tokens import (
    SEQUENCE_FORMAT_CONTROL_V1,
    build_control_primer,
    build_control_sequence,
    build_legacy_sequence,
    control_prefix_tokens,
    tempo_control_token,
    token_names,
)
from scripts.prepare_role_dataset import infer_tempo, notes_to_midi, write_tokenized_records
from utilities.constants import TOKEN_COND_SEP, TOKEN_END, TOKEN_TEMPO_DANCE, TOKEN_TEMPO_FAST


class ControlTokensTest(unittest.TestCase):
    def test_tempo_bucket_tokens_are_stable(self) -> None:
        self.assertEqual(tempo_control_token(124), TOKEN_TEMPO_DANCE)
        self.assertEqual(tempo_control_token(160), TOKEN_TEMPO_FAST)

    def test_control_sequence_uses_explicit_separator_not_end_token_between_prompt_and_target(self) -> None:
        seq = build_control_sequence([10, 11], [20, 21], role="lead", tempo_bpm=124)

        self.assertEqual(seq[:3], control_prefix_tokens("lead", 124))
        self.assertEqual(seq[5], TOKEN_COND_SEP)
        self.assertEqual(seq[-1], TOKEN_END)
        self.assertNotEqual(seq[5], TOKEN_END)

    def test_control_primer_keeps_prefix_when_truncated(self) -> None:
        primer = build_control_primer(list(range(20)), tempo_bpm=124, primer_max_tokens=6)

        self.assertEqual(primer[:3], control_prefix_tokens("lead", 124))
        self.assertEqual(primer[-1], TOKEN_COND_SEP)
        self.assertEqual(len(primer), 6)

    def test_legacy_sequence_remains_available(self) -> None:
        self.assertEqual(build_legacy_sequence([1], [2]), [1, TOKEN_END, 2, TOKEN_END])

    def test_token_names_reports_control_names(self) -> None:
        self.assertEqual(token_names(control_prefix_tokens("lead", 124)), ["ROLE_LEAD", "TEMPO_DANCE", "BAR"])

    def test_prepare_role_dataset_writes_control_v1_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sample = root / "000000"
            sample.mkdir()
            cond_path = sample / "conditioning.mid"
            target_path = sample / "target.mid"
            meta_path = sample / "meta.json"
            notes = [
                pretty_midi.Note(velocity=80, pitch=48, start=0.0, end=0.2),
                pretty_midi.Note(velocity=80, pitch=60, start=0.25, end=0.45),
            ]
            notes_to_midi(notes[:1], 124).write(str(cond_path))
            notes_to_midi(notes[1:], 124).write(str(target_path))
            meta_path.write_text(
                """
{
  "sample_id": "000000",
  "role": "lead",
  "tempo_bpm": 124.0
}
""".strip()
            )

            count = write_tokenized_records(
                [{"sample_id": "000000", "conditioning_path": cond_path, "target_path": target_path, "meta_path": meta_path}],
                "train",
                root / "tokenized",
                SEQUENCE_FORMAT_CONTROL_V1,
                "lead",
            )
            tokens = np.load(root / "tokenized" / "000000.npy")

        self.assertEqual(count, 1)
        self.assertEqual(tokens[:3].tolist(), control_prefix_tokens("lead", 124))
        self.assertIn(TOKEN_COND_SEP, tokens.tolist())

    def test_infer_tempo_reads_pretty_midi_tempo_values(self) -> None:
        pm = pretty_midi.PrettyMIDI(initial_tempo=124)

        self.assertAlmostEqual(infer_tempo(pm), 124.0)


if __name__ == "__main__":
    unittest.main()
