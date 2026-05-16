from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pretty_midi

from inference.app.conditioning import CONDITIONING_PITCH_MAX, build_request_conditioning_midi
from inference.app.fallback import chord_for_time
from inference.app.generator import candidate_quality_score, generate_midi_phrase
from inference.app.metrics import compute_midi_metrics, validate_metrics
from inference.app.schemas import GenerationMetrics, GenerationRequest


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
            chord_tone_ratio=None,
        )
        medium_candidate = SimpleNamespace(
            note_density=2.84,
            dead_air_ratio=0.50,
            repetition_score=0.0,
            chord_tone_ratio=None,
        )

        self.assertLess(
            candidate_quality_score(medium_candidate, "medium"),
            candidate_quality_score(sparse_candidate, "medium"),
        )

    def test_candidate_score_prefers_higher_chord_tone_ratio_when_other_metrics_match(self) -> None:
        low_chord_candidate = SimpleNamespace(
            note_density=3.0,
            dead_air_ratio=0.3,
            repetition_score=0.0,
            chord_tone_ratio=0.25,
        )
        high_chord_candidate = SimpleNamespace(
            note_density=3.0,
            dead_air_ratio=0.3,
            repetition_score=0.0,
            chord_tone_ratio=0.85,
        )

        self.assertLess(
            candidate_quality_score(high_chord_candidate, "medium"),
            candidate_quality_score(low_chord_candidate, "medium"),
        )

    def test_sparse_validation_allows_long_gaps_when_density_is_valid(self) -> None:
        metrics = GenerationMetrics(
            generation_time_ms=100,
            note_count=4,
            duration_sec=3.0,
            note_density=1.33,
            dead_air_ratio=1.0,
            repetition_score=0.0,
            pitch_min=60,
            pitch_max=72,
            fallback_used=False,
        )

        valid, reason = validate_metrics(metrics, "sparse")

        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_medium_validation_rejects_dead_air_at_threshold(self) -> None:
        metrics = GenerationMetrics(
            generation_time_ms=100,
            note_count=4,
            duration_sec=3.0,
            note_density=1.33,
            dead_air_ratio=0.8,
            repetition_score=0.0,
            pitch_min=60,
            pitch_max=72,
            fallback_used=False,
        )

        valid, reason = validate_metrics(metrics, "medium")

        self.assertFalse(valid)
        self.assertIn("dead-air ratio too high", str(reason))

    def test_chord_tone_metric_uses_note_start_time_pitch_class(self) -> None:
        request = GenerationRequest(
            bpm=120,
            chord_progression=["C7"],
            bars=1,
            density="medium",
            seed=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "chord_tone_metric.mid"
            pm = pretty_midi.PrettyMIDI(initial_tempo=float(request.bpm))
            piano = pretty_midi.Instrument(program=0, is_drum=False)
            for index, pitch in enumerate([60, 64, 67, 70, 61]):
                start = index * 0.2
                piano.notes.append(
                    pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start,
                        end=start + 0.1,
                    )
                )
            pm.instruments.append(piano)
            pm.write(str(output_path))

            metrics = compute_midi_metrics(
                output_path,
                generation_time_ms=100,
                fallback_used=False,
                request=request,
            )

        self.assertEqual(metrics.chord_tone_count, 4)
        self.assertEqual(metrics.non_chord_tone_count, 1)
        self.assertAlmostEqual(metrics.chord_tone_ratio or 0.0, 0.8)

    def test_generation_can_use_in_process_model_runner(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.calls = 0

            def generate_candidates(
                self,
                request: GenerationRequest,
                output_dir: str | Path,
                conditioning_midi: str | Path,
                primer_max_tokens: int,
                max_sequence: int,
                model_candidates: int,
            ) -> list[Path]:
                self.calls += 1
                raw_dir = Path(output_dir) / f"{request.job_id}_model_raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                output_path = raw_dir / "jazz_sample_1.mid"

                pm = pretty_midi.PrettyMIDI(initial_tempo=float(request.bpm))
                piano = pretty_midi.Instrument(program=0, is_drum=False)
                for index in range(12):
                    start = index * 0.1
                    piano.notes.append(
                        pretty_midi.Note(
                            velocity=80,
                            pitch=60 + (index % 5),
                            start=start,
                            end=start + 0.08,
                        )
                    )
                pm.instruments.append(piano)
                pm.write(str(output_path))
                return [output_path]

        request = GenerationRequest(
            bpm=120,
            chord_progression=["Cm7", "Fm7", "Bb7", "Ebmaj7"],
            bars=2,
            density="medium",
            job_id="unit_runner_path",
            seed=11,
        )
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_midi_phrase(
                request=request,
                output_dir=Path(tmp_dir),
                use_model=True,
                model_runner=runner,
            )

        self.assertEqual(runner.calls, 1)
        self.assertEqual(result.status, "COMPLETED")
        self.assertFalse(result.fallback_used)
        self.assertTrue(result.model_repaired)


if __name__ == "__main__":
    unittest.main()
