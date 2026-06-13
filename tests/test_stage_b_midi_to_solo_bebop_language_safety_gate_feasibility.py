import unittest

from scripts.run_stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep import (
    case_counts,
    filter_motion_rows,
    selectable_count,
    summarize_rows,
)


def row(case_label: str, *, step: float, chromatic: float, large_leap: float) -> dict:
    return {
        "case_label": case_label,
        "objective_metrics": {
            "step_motion_ratio": step,
            "chromatic_step_ratio": chromatic,
            "large_leap_ratio": large_leap,
            "adjacent_repeat_ratio": 0.0,
            "max_bar_pitch_class_jaccard": 0.62,
            "enclosure_proxy_ratio": 0.31,
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


if __name__ == "__main__":
    unittest.main()
