import unittest

from scripts.build_stage_b_midi_to_solo_bebop_language_best_of_package import (
    apply_rhythm_articulation_variation,
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


if __name__ == "__main__":
    unittest.main()
