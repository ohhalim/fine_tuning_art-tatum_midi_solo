from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from inference.app.model_runner import StageAModelRunner
from inference.app.schemas import GenerationRequest
from scripts.generate import _duration_seconds_to_steps
from scripts.run_resident_model_probe import (
    evaluate_generation_arm,
    parse_target_lengths,
    stage_a_musical_duration_ms,
    validate_generated_token_block,
)


class ResidentModelProbeTest(unittest.TestCase):
    def test_duration_target_rounds_up_to_avoid_bar_underfill(self) -> None:
        self.assertEqual(188, _duration_seconds_to_steps(1.875))
        self.assertEqual(177, _duration_seconds_to_steps(1.7641))
        with self.assertRaises(ValueError):
            _duration_seconds_to_steps(0)

    def test_generation_gate_stays_unknown_below_registered_sample_count(self) -> None:
        result = evaluate_generation_arm(
            target_total_tokens=64,
            primer_tokens=32,
            generated_token_counts=[32, 32],
            musical_durations_ms=[1900.0, 2000.0],
            generation_times_ms=[900.0, 1000.0],
            decode_validation_times_ms=[5.0, 5.0],
            lookahead_ms=1875.0,
            scheduling_margin_ms=20.0,
            minimum_gate_samples=20,
            one_bar_contract_validated=False,
        )

        self.assertFalse(result["timing_sample_count_sufficient"])
        self.assertFalse(result["measurement_sufficient_for_r2_gate"])
        self.assertIsNone(result["passed_r2_generation_deadline_gate"])

    def test_generation_gate_uses_p99_and_scheduling_margin(self) -> None:
        passing = evaluate_generation_arm(
            target_total_tokens=64,
            primer_tokens=32,
            generated_token_counts=[32] * 20,
            musical_durations_ms=[1900.0] * 20,
            generation_times_ms=[1000.0] * 20,
            decode_validation_times_ms=[5.0] * 20,
            lookahead_ms=1875.0,
            scheduling_margin_ms=20.0,
            minimum_gate_samples=20,
            one_bar_contract_validated=True,
        )
        failing = evaluate_generation_arm(
            target_total_tokens=128,
            primer_tokens=32,
            generated_token_counts=[96] * 20,
            musical_durations_ms=[1900.0] * 20,
            generation_times_ms=[1900.0] * 20,
            decode_validation_times_ms=[5.0] * 20,
            lookahead_ms=1875.0,
            scheduling_margin_ms=20.0,
            minimum_gate_samples=20,
            one_bar_contract_validated=True,
        )

        self.assertTrue(passing["passed_r2_generation_deadline_gate"])
        self.assertEqual(
            1005.0,
            passing["generation_p99_plus_decode_validation_p99_ms"],
        )
        self.assertTrue(passing["generation_decode_sample_counts_aligned"])
        self.assertFalse(failing["passed_r2_generation_deadline_gate"])

    def test_generation_gate_rejects_fast_output_that_does_not_cover_a_bar(self) -> None:
        result = evaluate_generation_arm(
            target_total_tokens=48,
            primer_tokens=32,
            generated_token_counts=[16] * 20,
            musical_durations_ms=[500.0] * 20,
            generation_times_ms=[400.0] * 20,
            decode_validation_times_ms=[5.0] * 20,
            lookahead_ms=1875.0,
            scheduling_margin_ms=20.0,
            minimum_gate_samples=20,
            one_bar_contract_validated=True,
        )

        self.assertFalse(result["all_samples_reach_target_duration"])
        self.assertEqual(0, result["samples_reaching_target_duration"])
        self.assertEqual(20, result["samples_under_target_duration"])
        self.assertEqual(500.0, result["generated_musical_duration_ms"]["minimum"])
        self.assertFalse(result["passed_r2_generation_deadline_gate"])

    def test_unvalidated_one_bar_contract_keeps_gate_unknown(self) -> None:
        result = evaluate_generation_arm(
            target_total_tokens=64,
            primer_tokens=32,
            generated_token_counts=[32] * 20,
            musical_durations_ms=[1900.0] * 20,
            generation_times_ms=[1000.0] * 20,
            decode_validation_times_ms=[5.0] * 20,
            lookahead_ms=1875.0,
            scheduling_margin_ms=20.0,
            minimum_gate_samples=20,
            one_bar_contract_validated=False,
        )

        self.assertTrue(result["timing_sample_count_sufficient"])
        self.assertFalse(result["measurement_sufficient_for_r2_gate"])
        self.assertIsNone(result["passed_r2_generation_deadline_gate"])

    def test_stage_a_duration_uses_time_shift_tokens_only(self) -> None:
        self.assertEqual(1010.0, stage_a_musical_duration_ms([60, 256, 355, 128]))

    def test_decoded_block_validation_accepts_exact_closed_one_bar(self) -> None:
        result = validate_generated_token_block(
            [376, 60, 355, 343, 188],
            lookahead_ms=1875.0,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(1880.0, result["musical_duration_ms"])
        self.assertEqual(0, result["orphan_note_off_count"])
        self.assertEqual(0, result["stuck_note_count"])
        self.assertLessEqual(result["decoded_note_end_max_ms"], 1880.0)

    def test_decoded_block_validation_rejects_underfill_and_orphan_note_off(self) -> None:
        result = validate_generated_token_block(
            [188, 60, 305, 188],
            lookahead_ms=1875.0,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["duration_matches_target"])
        self.assertEqual(1, result["orphan_note_off_count"])

    def test_target_length_parser_rejects_non_positive_values(self) -> None:
        self.assertEqual([48, 64], parse_target_lengths("48,64"))
        with self.assertRaises(Exception):
            parse_target_lengths("64,0")

    def test_stage_a_runner_enables_grammar_mask_by_default(self) -> None:
        runner = StageAModelRunner.__new__(StageAModelRunner)
        runner.model = object()
        runner.max_sequence = 64
        runner.grammar_mask = True
        request = GenerationRequest(
            bpm=128,
            chord_progression=["Dm7", "G7", "Cmaj7", "A7"],
            bars=1,
            seed=42,
            job_id="resident-test",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            conditioning = Path(temp_dir) / "conditioning.mid"
            conditioning.write_bytes(b"MThd")

            def write_midi(_tokens, output_path: str) -> None:
                Path(output_path).write_bytes(b"MThd")

            with (
                patch("scripts.generate.build_primer", return_value=torch.tensor([1, 2, 3])),
                patch("scripts.generate.generate_once", return_value=[4, 5]) as generate_once,
                patch("scripts.generate.decode_midi", side_effect=write_midi),
            ):
                candidates = runner.generate_candidates(
                    request=request,
                    output_dir=temp_dir,
                    conditioning_midi=conditioning,
                    primer_max_tokens=32,
                    max_sequence=64,
                    model_candidates=1,
                    control_format="control_v1",
                )

        self.assertEqual(1, len(candidates))
        self.assertTrue(generate_once.call_args.kwargs["grammar_mask"])
        self.assertEqual(1.875, generate_once.call_args.kwargs["target_duration_seconds"])


if __name__ == "__main__":
    unittest.main()
