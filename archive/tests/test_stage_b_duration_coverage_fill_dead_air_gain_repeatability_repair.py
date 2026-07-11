from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.summarize_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair import (
    StageBDurationCoverageDeadAirGainRepeatabilityRepairError,
    build_dead_air_gain_repeatability_repair_report,
    validate_dead_air_gain_repeatability_repair,
)
from tests.test_stage_b_duration_coverage_fill_broader_repeatability_sweep import (
    distinct_sample_seed_sweep,
    write_source_run,
)


def broader_repeatability_sweep(*, boundary: str = "qualified_gate_repeatability_with_partial_dead_air_gain") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_broader_repeatability_sweep_v1",
        "repeatability_summary": {
            "boundary": boundary,
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "dead_air_improved_source_candidate_count": 1,
            "broad_model_quality_claimed": False,
        },
    }


def dead_air_gain_distinct_sample_seed_sweep() -> dict:
    sweep = distinct_sample_seed_sweep()
    for candidate in sweep["top_candidates"]:
        candidate["dead_air_ratio"] = 0.5714285714285714
    return sweep


class StageBDurationCoverageFillDeadAirGainRepeatabilityRepairTest(unittest.TestCase):
    def test_repairs_selection_to_uniform_dead_air_gain(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generation_root = root / "outputs" / "stage_b_generation_probe"
            write_source_run(generation_root, "run_a", 1)
            write_source_run(generation_root, "run_b", 2)

            report = build_dead_air_gain_repeatability_repair_report(
                broader_repeatability_sweep=broader_repeatability_sweep(),
                distinct_sample_seed_sweep=dead_air_gain_distinct_sample_seed_sweep(),
                output_dir=root / "repair",
                generation_output_root=generation_root,
                max_source_candidates=2,
                min_source_candidates=2,
                min_dead_air_gain_source_candidates=2,
                fill_max_additions=[4, 6, 8, 10],
                dead_air_threshold_sec=0.18,
                simultaneous_limit=1,
                min_unique_pitch_count=7,
                max_dead_air_ratio_exclusive=0.376,
                min_note_count=12,
                max_simultaneous_notes=1,
                max_duplicated_3_note_chunks=0,
                max_adjacent_pitch_repeats_exclusive=1,
                max_interval_exclusive=12,
            )
            summary = validate_dead_air_gain_repeatability_repair(
                report,
                expected_boundary="qualified_gate_repeatability_with_dead_air_gain",
                min_source_candidates=2,
                min_dead_air_gain_source_candidates=2,
                require_no_broad_quality_claim=True,
            )

            self.assertEqual(summary["source_candidate_count"], 2)
            self.assertEqual(summary["qualified_source_candidate_count"], 2)
            self.assertEqual(summary["dead_air_gain_source_candidate_count"], 2)
            self.assertEqual(summary["total_variant_count"], 8)
            self.assertEqual(summary["selected_fill_additions"], [6])
            self.assertFalse(summary["broad_model_quality_claimed"])
            for result in report["source_repeatability_results"]:
                self.assertTrue(result["dead_air_gain_repaired"])
                self.assertLess(result["selected_dead_air_ratio"], result["baseline_dead_air_ratio"])

    def test_rejects_unexpected_previous_boundary(self) -> None:
        with self.assertRaises(StageBDurationCoverageDeadAirGainRepeatabilityRepairError):
            build_dead_air_gain_repeatability_repair_report(
                broader_repeatability_sweep=broader_repeatability_sweep(boundary="other_boundary"),
                distinct_sample_seed_sweep=distinct_sample_seed_sweep(),
                output_dir=Path("outputs/repair"),
                generation_output_root=Path("missing"),
                max_source_candidates=2,
                min_source_candidates=2,
                min_dead_air_gain_source_candidates=2,
                fill_max_additions=[4, 6, 8, 10],
                dead_air_threshold_sec=0.18,
                simultaneous_limit=1,
                min_unique_pitch_count=7,
                max_dead_air_ratio_exclusive=0.376,
                min_note_count=12,
                max_simultaneous_notes=1,
                max_duplicated_3_note_chunks=0,
                max_adjacent_pitch_repeats_exclusive=1,
                max_interval_exclusive=12,
            )


if __name__ == "__main__":
    unittest.main()
