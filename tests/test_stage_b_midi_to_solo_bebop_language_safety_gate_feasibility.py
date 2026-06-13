import unittest

from scripts.run_stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep import (
    build_guard_motion_configs,
    case_counts,
    filter_guard_rows,
    filter_motion_rows,
    selectable_count,
    summarize_feasible_configs,
    summarize_rows,
)


def row(
    case_label: str,
    *,
    step: float,
    chromatic: float,
    large_leap: float,
    offbeat: float = 0.390625,
    bar_similarity: float = 0.62,
) -> dict:
    return {
        "gate_penalty": 0.0,
        "case_label": case_label,
        "objective_metrics": {
            "step_motion_ratio": step,
            "chromatic_step_ratio": chromatic,
            "large_leap_ratio": large_leap,
            "adjacent_repeat_ratio": 0.0,
            "max_bar_pitch_class_jaccard": bar_similarity,
            "enclosure_proxy_ratio": 0.31,
            "offbeat_non_chord_ratio": offbeat,
            "offbeat_unresolved_non_chord_ratio": 0.0,
            "dominant_altered_offbeat_ratio": 0.125,
        },
    }


class BebopLanguageSafetyGateFeasibilityTest(unittest.TestCase):
    def test_selectable_count_caps_each_case(self) -> None:
        counts = {"dominant_cycle": 3, "major_ii_v_turnaround": 1, "rhythm_turnaround": 5}

        self.assertEqual(selectable_count(counts, max_per_case=2), 5)
        self.assertEqual(selectable_count(counts, max_per_case=3), 7)

    def test_motion_filter_requires_all_thresholds(self) -> None:
        safe = row("dominant_cycle", step=0.40, chromatic=0.22, large_leap=0.05)
        low_step = row("dominant_cycle", step=0.34, chromatic=0.22, large_leap=0.05)
        low_chromatic = row("minor_backdoor", step=0.40, chromatic=0.16, large_leap=0.05)
        high_leap = row("rhythm_turnaround", step=0.40, chromatic=0.22, large_leap=0.12)

        filtered = filter_motion_rows(
            [safe, low_step, low_chromatic, high_leap],
            min_step_motion_ratio=0.38,
            min_chromatic_step_ratio=0.20,
            max_large_leap_ratio=0.08,
        )

        self.assertEqual(filtered, [safe])

    def test_summary_reports_case_counts_and_metric_averages(self) -> None:
        rows = [
            row("dominant_cycle", step=0.40, chromatic=0.22, large_leap=0.04),
            row("dominant_cycle", step=0.38, chromatic=0.20, large_leap=0.08),
            row("minor_backdoor", step=0.36, chromatic=0.18, large_leap=0.06),
        ]

        summary = summarize_rows(rows, max_per_case=1)

        self.assertEqual(summary["case_counts"], {"dominant_cycle": 2, "minor_backdoor": 1})
        self.assertEqual(summary["selectable_count"], 2)
        self.assertAlmostEqual(summary["avg_step_motion_ratio"], 0.38)

    def test_guard_filter_applies_offbeat_and_bar_similarity_caps(self) -> None:
        safe = row("dominant_cycle", step=0.40, chromatic=0.22, large_leap=0.04)
        high_offbeat = row("minor_backdoor", step=0.40, chromatic=0.22, large_leap=0.04, offbeat=0.40625)
        high_bar_similarity = row(
            "rhythm_turnaround",
            step=0.40,
            chromatic=0.22,
            large_leap=0.04,
            bar_similarity=0.68,
        )

        filtered = filter_guard_rows(
            [safe, high_offbeat, high_bar_similarity],
            max_gate_penalty=0.0,
            max_offbeat_non_chord_ratio=0.390625,
            max_unresolved_offbeat_non_chord_ratio=0.0,
            max_dominant_altered_offbeat_ratio=0.25,
            max_adjacent_repeat_ratio=0.0,
            max_bar_pitch_class_jaccard=0.65,
        )

        self.assertEqual(filtered, [safe])

    def test_guard_motion_configs_report_selected_count_feasibility(self) -> None:
        rows = [
            row("dominant_cycle", step=0.42, chromatic=0.24, large_leap=0.04),
            row("dominant_cycle", step=0.41, chromatic=0.23, large_leap=0.04),
            row("minor_backdoor", step=0.40, chromatic=0.22, large_leap=0.05),
            row("rhythm_turnaround", step=0.40, chromatic=0.22, large_leap=0.05, offbeat=0.40625),
        ]

        configs = build_guard_motion_configs(
            rows,
            offbeat_values=[0.390625, 0.40625],
            bar_similarity_values=[0.65],
            step_values=[0.40],
            chromatic_values=[0.22],
            large_leap_values=[0.055],
            max_per_case_values=[1, 2],
            selected_count=4,
            max_gate_penalty=0.0,
            max_unresolved_offbeat_non_chord_ratio=0.0,
            max_dominant_altered_offbeat_ratio=0.25,
            max_adjacent_repeat_ratio=0.0,
        )

        strict_offbeat = next(item for item in configs if item["max_offbeat_non_chord_ratio"] == 0.390625)
        relaxed_offbeat = next(item for item in configs if item["max_offbeat_non_chord_ratio"] == 0.40625)
        self.assertFalse(strict_offbeat["feasible_for_selected_count"]["1"])
        self.assertFalse(strict_offbeat["feasible_for_selected_count"]["2"])
        self.assertFalse(relaxed_offbeat["feasible_for_selected_count"]["1"])
        self.assertTrue(relaxed_offbeat["feasible_for_selected_count"]["2"])
        self.assertGreater(
            relaxed_offbeat["candidate_count"],
            strict_offbeat["candidate_count"],
        )

        summary = summarize_feasible_configs(
            configs,
            baseline_max_offbeat_non_chord_ratio=0.40625,
            baseline_max_bar_pitch_class_jaccard=0.70,
            max_per_case_values=[1, 2],
        )
        self.assertEqual(summary["min_feasible_max_offbeat_non_chord_ratio"], 0.40625)
        self.assertEqual(summary["min_feasible_max_bar_pitch_class_jaccard"], 0.65)
        self.assertFalse(summary["stricter_offbeat_feasible"])
        self.assertTrue(summary["stricter_bar_similarity_feasible"])


if __name__ == "__main__":
    unittest.main()
