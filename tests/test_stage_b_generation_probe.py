from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi
import torch

from scripts.run_stage_b_generation_probe import (
    analyze_stage_b_note_grammar,
    build_stage_b_primer,
    dedupe_and_limit_notes,
    decode_tokens_to_midi,
    generate_stage_b_constrained_tokens,
    generate_stage_b_tokens,
    postprocess_stage_b_midi,
)
from scripts.stage_b_tokens import (
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    position_token,
)
from utilities.constants import TOKEN_BAR, TOKEN_END, TOKEN_ROLE_LEAD, VOCAB_SIZE


class FakeStageBModel:
    def __init__(self, returned_tokens: list[int]) -> None:
        self.returned_tokens = returned_tokens
        self.sample_vocab_size: int | None = None

    def generate(self, **kwargs):
        self.sample_vocab_size = kwargs["sample_vocab_size"]
        return torch.tensor([self.returned_tokens], dtype=torch.long)


class FakeConstrainedModel:
    def __call__(self, tokens):
        return torch.zeros((1, tokens.shape[1], VOCAB_SIZE), dtype=torch.float32)


class StageBGenerationProbeTest(unittest.TestCase):
    def test_build_stage_b_primer_contains_bar_and_first_chord(self) -> None:
        primer = build_stage_b_primer(["Cm7", "F7"], bpm=124)

        self.assertEqual(primer[0], TOKEN_ROLE_LEAD)
        self.assertEqual(primer[2], TOKEN_BAR)
        self.assertEqual(primer[3:5], chord_tokens("Cm7"))

    def test_generation_uses_full_stage_b_vocab_size(self) -> None:
        returned_tokens = build_stage_b_primer(["Cm7"], bpm=124) + [TOKEN_END]
        model = FakeStageBModel(returned_tokens)

        tokens = generate_stage_b_tokens(
            model=model,
            primer_tokens=returned_tokens[:-1],
            target_length=16,
            temperature=0.9,
            top_k=32,
            top_p=None,
        )

        self.assertEqual(tokens, returned_tokens)
        self.assertEqual(model.sample_vocab_size, VOCAB_SIZE)

    def test_decode_tokens_to_midi_writes_stage_b_notes(self) -> None:
        tokens = build_stage_b_primer(["Cm7"], bpm=120) + [
            position_token(0),
            note_velocity_token(4),
            note_pitch_token(60),
            note_duration_token(2),
            TOKEN_END,
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = Path(tmp_dir) / "decoded.mid"

            decode_tokens_to_midi(tokens, midi_path, bpm=120)

            midi = pretty_midi.PrettyMIDI(str(midi_path))
            notes = midi.instruments[0].notes
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].pitch, 60)
            self.assertAlmostEqual(notes[0].start, 0.0)
            self.assertAlmostEqual(notes[0].end, 0.25)

    def test_analyze_stage_b_note_grammar_counts_complete_groups(self) -> None:
        primer = build_stage_b_primer(["Cm7"], bpm=120)
        tokens = primer + [
            position_token(0),
            note_velocity_token(4),
            note_pitch_token(60),
            note_duration_token(2),
            TOKEN_END,
        ]

        report = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))

        self.assertEqual(report["complete_note_groups"], 1)
        self.assertEqual(report["invalid_token_count"], 0)
        self.assertTrue(report["grammar_valid"])

    def test_analyze_stage_b_note_grammar_reports_incomplete_groups(self) -> None:
        primer = build_stage_b_primer(["Cm7"], bpm=120)
        tokens = primer + [position_token(0), note_pitch_token(60), TOKEN_END]

        report = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))

        self.assertEqual(report["complete_note_groups"], 0)
        self.assertGreater(report["invalid_token_count"], 0)
        self.assertFalse(report["grammar_valid"])

    def test_constrained_generation_creates_decodable_note_groups(self) -> None:
        primer = build_stage_b_primer(["Cm7", "F7"], bpm=120)

        tokens = generate_stage_b_constrained_tokens(
            model=FakeConstrainedModel(),
            primer_tokens=primer,
            chords=["Cm7", "F7"],
            bpm=120,
            bars=2,
            note_groups_per_bar=1,
            max_sequence=64,
            temperature=1.0,
            top_k=1,
        )

        report = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        self.assertEqual(report["complete_note_groups"], 2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = Path(tmp_dir) / "constrained.mid"
            decode_tokens_to_midi(tokens, midi_path, bpm=120)
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            self.assertEqual(len(midi.instruments[0].notes), 2)

    def test_dedupe_and_limit_notes_removes_same_onset_pitch_duplicates(self) -> None:
        notes = [
            pretty_midi.Note(velocity=64, pitch=60, start=0.0, end=0.25),
            pretty_midi.Note(velocity=96, pitch=60, start=0.0, end=0.5),
            pretty_midi.Note(velocity=80, pitch=64, start=0.0, end=0.25),
        ]

        processed = dedupe_and_limit_notes(notes, simultaneous_limit=2)

        self.assertEqual(len(processed), 2)
        self.assertEqual([note.pitch for note in processed], [60, 64])
        self.assertEqual(processed[0].velocity, 96)

    def test_postprocess_stage_b_midi_limits_simultaneous_notes(self) -> None:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0, is_drum=False)
        piano.notes.extend(
            [
                pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
                pretty_midi.Note(velocity=82, pitch=64, start=0.0, end=0.5),
                pretty_midi.Note(velocity=84, pitch=67, start=0.0, end=0.5),
                pretty_midi.Note(velocity=86, pitch=72, start=0.5, end=0.75),
            ]
        )
        midi.instruments.append(piano)

        report = postprocess_stage_b_midi(midi, simultaneous_limit=2)

        self.assertEqual(report["before_note_count"], 4)
        self.assertEqual(report["after_note_count"], 3)
        self.assertEqual(report["before_max_simultaneous_notes"], 3)
        self.assertEqual(report["after_max_simultaneous_notes"], 2)
        self.assertEqual(len(midi.instruments[0].notes), 3)


if __name__ == "__main__":
    unittest.main()
