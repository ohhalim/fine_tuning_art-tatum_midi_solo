import unittest

from scripts.build_stage_b_midi_to_solo_bebop_language_best_of_package import (
    apply_rhythm_articulation_variation,
    bebop_stepwise_chromatic_selection_score,
    filter_candidate_rows,
    rhythm_articulation_metrics,
)
from scripts.build_stage_b_midi_to_solo_bebop_language_package import (
    build_bebop_candidate,
    candidate_gate_penalty,
    objective_metrics,
)


class BebopLanguageRhythmArticulationTest(unittest.TestCase):
    def test_reduces_duration_template_repeat_without_gate_regression(self) -> None:
        chords = ["Dm7", "G7", "Cmaj7", "A7"]
        pm, _meta = build_bebop_candidate(
            chords,
            seed=17,
            bars=8,
            bpm=124.0,
            non_chord_probability=0.38,
        )

        before_rhythm = rhythm_articulation_metrics(pm, bars=8, bpm=124.0)
        before_objective = objective_metrics(pm, chords, bars=8, bpm=124.0)
        before_gate = candidate_gate_penalty(before_objective)

        repaired, report = apply_rhythm_articulation_variation(pm, bars=8, bpm=124.0)
        after_rhythm = rhythm_articulation_metrics(repaired, bars=8, bpm=124.0)
        after_objective = objective_metrics(repaired, chords, bars=8, bpm=124.0)
        after_gate = candidate_gate_penalty(after_objective)

        self.assertTrue(report["attempted"])
        self.assertTrue(report["changed"])
        self.assertLess(
            after_rhythm["duration_template_repeat_ratio"],
            before_rhythm["duration_template_repeat_ratio"],
        )
        self.assertGreater(
            after_rhythm["unique_duration_bucket_count"],
            before_rhythm["unique_duration_bucket_count"],
        )
        self.assertGreater(
            after_rhythm["unique_velocity_count"],
            before_rhythm["unique_velocity_count"],
        )
        self.assertLessEqual(after_gate, before_gate)
        self.assertEqual(after_objective["note_count"], before_objective["note_count"])
        self.assertEqual(after_objective["adjacent_repeat_ratio"], before_objective["adjacent_repeat_ratio"])

    def test_stepwise_chromatic_profile_prefers_stepwise_candidate(self) -> None:
        base_metrics = {
            "gate_penalty": 0.0,
            "offbeat_unresolved_non_chord_ratio": 0.0,
            "offbeat_non_chord_resolution_ratio": 1.0,
            "large_leap_ratio": 0.05,
            "max_abs_interval": 7,
            "enclosure_proxy_ratio": 0.34,
            "max_bar_pitch_class_jaccard": 0.58,
            "adjacent_repeat_ratio": 0.0,
            "two_note_cycle_ratio": 0.0,
            "interval_trigram_repeat_ratio": 0.0,
            "bar_half_repeat_ratio": 0.0,
            "offbeat_non_chord_ratio": 0.390625,
            "chord_tone_ratio": 0.8046875,
            "dominant_altered_offbeat_ratio": 0.125,
            "unique_pitch_count": 14,
            "step_motion_ratio": 0.34,
            "chromatic_step_ratio": 0.16,
            "third_fourth_motion_ratio": 0.60,
        }
        arpeggio_like = {
            "objective_metrics": base_metrics,
            "score": 0.10,
            "gate_penalty": 0.0,
        }
        stepwise_like = {
            "objective_metrics": {
                **base_metrics,
                "step_motion_ratio": 0.42,
                "chromatic_step_ratio": 0.24,
                "third_fourth_motion_ratio": 0.52,
            },
            "score": 0.10,
            "gate_penalty": 0.0,
        }

        self.assertLess(
            bebop_stepwise_chromatic_selection_score(stepwise_like),
            bebop_stepwise_chromatic_selection_score(arpeggio_like),
        )

    def test_safety_filter_excludes_adjacent_repeat_and_bar_similarity_regressions(self) -> None:
        base_metrics = {
            "offbeat_non_chord_ratio": 0.390625,
            "offbeat_unresolved_non_chord_ratio": 0.0,
            "dominant_altered_offbeat_ratio": 0.125,
            "adjacent_repeat_ratio": 0.0,
            "max_bar_pitch_class_jaccard": 0.62,
        }
        safe = {
            "gate_penalty": 0.0,
            "objective_metrics": base_metrics,
        }
        adjacent_repeat_regression = {
            "gate_penalty": 0.0,
            "objective_metrics": {
                **base_metrics,
                "adjacent_repeat_ratio": 0.02,
            },
        }
        bar_similarity_regression = {
            "gate_penalty": 0.0,
            "objective_metrics": {
                **base_metrics,
                "max_bar_pitch_class_jaccard": 0.72,
            },
        }

        filtered = filter_candidate_rows(
            [safe, adjacent_repeat_regression, bar_similarity_regression],
            max_gate_penalty=0.0,
            max_offbeat_non_chord_ratio=0.40625,
            max_unresolved_offbeat_non_chord_ratio=0.0,
            max_dominant_altered_offbeat_ratio=0.25,
            max_adjacent_repeat_ratio=0.0,
            max_bar_pitch_class_jaccard=0.625,
        )

        self.assertEqual(filtered, [safe])


if __name__ == "__main__":
    unittest.main()
