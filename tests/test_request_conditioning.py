from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pretty_midi

from inference.app.conditioning import CONDITIONING_PITCH_MAX, build_request_conditioning_midi
from inference.app.fallback import chord_for_time
from inference.app.generator import candidate_quality_score
from inference.app.schemas import GenerationRequest


class RequestConditioningTest(unittest.TestCase):
    def test_chord_progression_is_spread_across_phrase_duration(self) -> None:
        request = GenerationRequest(
            bpm=120,
            chord_progression=["Cm7", "Fm7", "Bb7", "Ebmaj7"],
            bars=2,
            seed=11,
        )

        self.assertEqual(chord_for_time(request, 0.00), "Cm7")
        self.assertEqual(chord_for_time(request, 1.01), "Fm7")
        self.assertEqual(chord_for_time(request, 2.01), "Bb7")
        self.assertEqual(chord_for_time(request, 3.01), "Ebmaj7")

    def test_request_conditioning_midi_contains_low_register_chord_guide(self) -> None:
        request = GenerationRequest(
            bpm=120,
            chord_progression=["Cm7", "Fm7", "Bb7", "Ebmaj7"],
            bars=2,
            job_id="unit_request_conditioning",
            seed=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = build_request_conditioning_midi(request, Path(tmp_dir))
            pm = pretty_midi.PrettyMIDI(str(output_path))

        self.assertEqual(len(pm.instruments), 1)
        notes = pm.instruments[0].notes
        self.assertGreaterEqual(len(notes), 12)
        self.assertTrue(all(note.pitch <= CONDITIONING_PITCH_MAX for note in notes))

        starts = sorted({round(float(note.start), 1) for note in notes})
        self.assertEqual(starts, [0.0, 1.0, 2.0, 3.0])

    def test_sampling_parameters_are_validated(self) -> None:
        request = GenerationRequest(
            bpm=120,
            chord_progression=["Cm7"],
            temperature=0.0,
        )

        with self.assertRaisesRegex(ValueError, "temperature"):
            request.validate()

    def test_candidate_score_penalizes_density_target_miss(self) -> None:
        sparse_candidate = SimpleNamespace(
            note_density=1.03,
            dead_air_ratio=0.33,
            repetition_score=0.0,
        )
        medium_candidate = SimpleNamespace(
            note_density=2.84,
            dead_air_ratio=0.50,
            repetition_score=0.0,
        )

        self.assertLess(
            candidate_quality_score(medium_candidate, "medium"),
            candidate_quality_score(sparse_candidate, "medium"),
        )


if __name__ == "__main__":
    unittest.main()
