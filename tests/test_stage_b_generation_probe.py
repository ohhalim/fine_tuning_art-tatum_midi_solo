from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi
import torch

from scripts.run_stage_b_generation_probe import (
    analyze_stage_b_collapse,
    analyze_stage_b_note_grammar,
    analyze_stage_b_temporal_coverage,
    build_probe_summary,
    build_stage_b_primer,
    coverage_aware_position_tokens,
    dedupe_and_limit_notes,
    decode_tokens_to_midi,
    evaluate_collapse_gate,
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
    position_from_token,
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

    def test_coverage_aware_constrained_generation_spreads_positions(self) -> None:
        primer = build_stage_b_primer(["Cm7"], bpm=120)

        tokens = generate_stage_b_constrained_tokens(
            model=FakeConstrainedModel(),
            primer_tokens=primer,
            chords=["Cm7"],
            bpm=120,
            bars=1,
            note_groups_per_bar=4,
            max_sequence=64,
            temperature=1.0,
            top_k=1,
            coverage_aware_positions=True,
        )

        coverage = analyze_stage_b_temporal_coverage(tokens, primer_size=len(primer), bars=1)

        self.assertEqual(coverage["per_bar_unique_onset_positions"], {"0": 4})
        self.assertEqual(coverage["earliest_absolute_position"], 0)
        self.assertEqual(coverage["latest_absolute_position"], 9)

    def test_coverage_aware_position_tokens_spread_onset_pairs(self) -> None:
        positions = [
            position_from_token(coverage_aware_position_tokens(index, note_groups_per_bar=4)[0])
            for index in range(4)
        ]

        self.assertEqual(positions, [0, 1, 8, 9])

    def test_coverage_aware_position_tokens_can_include_small_window(self) -> None:
        positions = [
            position_from_token(token)
            for token in coverage_aware_position_tokens(2, note_groups_per_bar=4, position_window=1)
        ]

        self.assertEqual(positions, [7, 8, 9])

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

    def test_build_probe_summary_tracks_multisample_thresholds(self) -> None:
        rows = [
            {"sample_index": 1, "valid": True, "grammar_gate_passed": True},
            {"sample_index": 2, "valid": False, "grammar_gate_passed": True, "failure_reason": "too sparse"},
            {"sample_index": 3, "valid": True, "grammar_gate_passed": True},
        ]

        summary = build_probe_summary(rows, min_valid_samples=2, require_all_grammar_samples=True)

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 2)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["valid_sample_indices"], [1, 3])
        self.assertTrue(summary["passed_generation_gate"])
        self.assertTrue(summary["passed_grammar_gate"])
        self.assertEqual(summary["failure_reasons"], {"too sparse": 1})

    def test_build_probe_summary_can_require_all_grammar_samples(self) -> None:
        rows = [
            {"sample_index": 1, "valid": True, "grammar_gate_passed": True},
            {"sample_index": 2, "valid": True, "grammar_gate_passed": False},
        ]

        summary = build_probe_summary(rows, min_valid_samples=1, require_all_grammar_samples=True)

        self.assertTrue(summary["passed_generation_gate"])
        self.assertFalse(summary["passed_grammar_gate"])

    def test_analyze_stage_b_collapse_flags_repeated_position_pitch_pairs(self) -> None:
        primer = build_stage_b_primer(["Cm7"], bpm=120)
        tokens = primer + [
            position_token(2),
            note_velocity_token(4),
            note_pitch_token(68),
            note_duration_token(2),
            position_token(2),
            note_velocity_token(4),
            note_pitch_token(68),
            note_duration_token(2),
            position_token(2),
            note_velocity_token(4),
            note_pitch_token(68),
            note_duration_token(2),
            TOKEN_END,
        ]

        report = analyze_stage_b_collapse(
            tokens,
            primer_size=len(primer),
            postprocess_report={"before_note_count": 3, "removed_note_count": 2},
        )

        self.assertEqual(report["note_group_count"], 3)
        self.assertEqual(report["unique_pitch_count"], 1)
        self.assertEqual(report["unique_position_count"], 1)
        self.assertEqual(report["unique_position_pitch_pair_count"], 1)
        self.assertAlmostEqual(report["repeated_position_pitch_pair_ratio"], 2 / 3)
        self.assertAlmostEqual(report["postprocess_removal_ratio"], 2 / 3)
        self.assertTrue(report["collapse_warning"])
        self.assertIn("repeated_position_pitch", report["collapse_reasons"])
        self.assertIn("postprocess_removed_majority", report["collapse_reasons"])

    def test_analyze_stage_b_temporal_coverage_reports_empty_spans(self) -> None:
        primer = build_stage_b_primer(["Cm7", "F7"], bpm=120)
        tokens = primer + [
            position_token(0),
            note_velocity_token(4),
            note_pitch_token(60),
            note_duration_token(2),
            position_token(8),
            note_velocity_token(4),
            note_pitch_token(64),
            note_duration_token(2),
            TOKEN_BAR,
            *chord_tokens("F7"),
            position_token(12),
            note_velocity_token(4),
            note_pitch_token(67),
            note_duration_token(4),
            TOKEN_END,
        ]

        report = analyze_stage_b_temporal_coverage(tokens, primer_size=len(primer), bars=2)

        self.assertEqual(report["note_group_count"], 3)
        self.assertEqual(report["unique_onset_position_count"], 3)
        self.assertAlmostEqual(report["onset_coverage_ratio"], 3 / 32)
        self.assertAlmostEqual(report["sustained_coverage_ratio"], 8 / 32)
        self.assertEqual(report["earliest_absolute_position"], 0)
        self.assertEqual(report["latest_absolute_position"], 28)
        self.assertEqual(report["position_span_steps"], 29)
        self.assertEqual(report["tail_empty_steps"], 3)
        self.assertEqual(report["per_bar_unique_onset_positions"], {"0": 2, "1": 1})

    def test_build_probe_summary_aggregates_collapse_diagnostics(self) -> None:
        rows = [
            {
                "sample_index": 1,
                "valid": False,
                "grammar_gate_passed": True,
                "failure_reason": "note count too low",
                "diagnostic_failure_reason": "note count too low; collapse=repeated_position_pitch",
                "collapse": {
                    "collapse_warning": True,
                    "repeated_position_pitch_pair_ratio": 0.75,
                    "postprocess_removal_ratio": 0.5,
                },
                "temporal_coverage": {
                    "onset_coverage_ratio": 0.10,
                    "sustained_coverage_ratio": 0.20,
                    "position_span_ratio": 0.50,
                    "longest_sustained_empty_run_steps": 8,
                },
            },
            {
                "sample_index": 2,
                "valid": True,
                "grammar_gate_passed": True,
                "collapse": {
                    "collapse_warning": False,
                    "repeated_position_pitch_pair_ratio": 0.25,
                    "postprocess_removal_ratio": 0.0,
                },
                "temporal_coverage": {
                    "onset_coverage_ratio": 0.20,
                    "sustained_coverage_ratio": 0.30,
                    "position_span_ratio": 0.75,
                    "longest_sustained_empty_run_steps": 4,
                },
            },
        ]

        summary = build_probe_summary(rows, min_valid_samples=1, require_all_grammar_samples=True)

        self.assertEqual(summary["collapse_warning_sample_count"], 1)
        self.assertAlmostEqual(summary["collapse_warning_sample_rate"], 0.5)
        self.assertAlmostEqual(summary["avg_repeated_position_pitch_pair_ratio"], 0.5)
        self.assertAlmostEqual(summary["max_postprocess_removal_ratio"], 0.5)
        self.assertAlmostEqual(summary["avg_onset_coverage_ratio"], 0.15)
        self.assertAlmostEqual(summary["avg_sustained_coverage_ratio"], 0.25)
        self.assertAlmostEqual(summary["avg_position_span_ratio"], 0.625)
        self.assertEqual(summary["max_longest_sustained_empty_run_steps"], 8)
        self.assertEqual(
            summary["diagnostic_failure_reasons"],
            {"note count too low; collapse=repeated_position_pitch": 1},
        )

    def test_evaluate_collapse_gate_reports_strict_failures(self) -> None:
        collapse = {
            "unique_pitch_count": 1,
            "unique_position_count": 1,
            "unique_position_pitch_pair_count": 1,
            "repeated_position_pitch_pair_ratio": 0.75,
            "postprocess_removal_ratio": 0.5,
        }

        gate = evaluate_collapse_gate(collapse)

        self.assertFalse(gate["passed"])
        self.assertIn("unique pitch count too low: 1 < 3", gate["failure_reasons"])
        self.assertIn(
            "repeated position/pitch pair ratio too high: 0.750 > 0.490",
            gate["failure_reasons"],
        )

    def test_build_probe_summary_tracks_strict_review_gate(self) -> None:
        rows = [
            {
                "sample_index": 1,
                "valid": True,
                "strict_valid": True,
                "grammar_gate_passed": True,
                "collapse": {
                    "collapse_warning": False,
                    "repeated_position_pitch_pair_ratio": 0.25,
                    "postprocess_removal_ratio": 0.0,
                },
            },
            {
                "sample_index": 2,
                "valid": False,
                "strict_valid": False,
                "grammar_gate_passed": True,
                "failure_reason": "note count too low",
                "collapse": {
                    "collapse_warning": True,
                    "repeated_position_pitch_pair_ratio": 0.75,
                    "postprocess_removal_ratio": 0.5,
                },
            },
        ]

        summary = build_probe_summary(
            rows,
            min_valid_samples=1,
            min_strict_valid_samples=1,
            require_all_grammar_samples=True,
            max_collapse_warning_sample_rate=0.5,
        )

        self.assertEqual(summary["strict_valid_sample_count"], 1)
        self.assertEqual(summary["strict_valid_sample_indices"], [1])
        self.assertTrue(summary["passed_strict_generation_gate"])
        self.assertTrue(summary["passed_collapse_rate_gate"])
        self.assertTrue(summary["passed_strict_review_gate"])
        self.assertEqual(
            summary["strict_failure_reasons"],
            {"midi_review_gate_failed: note count too low": 1},
        )


if __name__ == "__main__":
    unittest.main()
