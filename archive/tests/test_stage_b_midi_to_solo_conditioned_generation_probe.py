from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_conditioned_generation_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloConditionedGenerationProbeError,
    build_conditioned_generation_report,
    validate_conditioned_generation_report,
)


def context_report() -> dict:
    bars = [
        ("C", "maj7", 36, "pitch_class_inference", 0.9),
        ("F", "dom7", 41, "pitch_class_inference", 0.9),
        ("G", "dom7", 43, "pitch_class_inference", 0.9),
        ("C", "maj7", 36, "pitch_class_inference", 0.9),
        ("C", "maj7", None, "carry_forward_empty_bar", 0.45),
        ("C", "maj7", None, "carry_forward_empty_bar", 0.45),
        ("C", "maj7", None, "carry_forward_empty_bar", 0.45),
        ("C", "maj7", None, "carry_forward_empty_bar", 0.45),
    ]
    return {
        "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
        "summary": {
            "context_bars": 8,
            "positions_per_bar": 16,
            "context_event_count": 128,
            "inferred_chord_bar_count": 4,
            "carry_forward_chord_bar_count": 4,
            "unknown_chord_bar_count": 0,
            "low_confidence_bar_count": 4,
            "bass_note_bar_count": 4,
        },
        "context": {
            "bar_contexts": [
                {
                    "bar_index": index,
                    "chord_root": root,
                    "chord_quality": quality,
                    "bass_note": bass,
                    "chord_source": source,
                    "chord_confidence": confidence,
                }
                for index, (root, quality, bass, source, confidence) in enumerate(bars)
            ]
        },
        "readiness": {
            "context_extraction_completed": True,
            "midi_to_solo_mvp_claimed": False,
            "harmony_analysis_quality_claimed": False,
        },
    }


def resource_probe(*, ready: bool = True, final_claim: bool = False) -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_training_resource_probe",
        "context_resource": {"context_event_count": 128},
        "full_window_resource": {"tokenized_train_files": 154136},
        "scale_smoke_resource": {"checkpoint_count": 1},
        "readiness": {
            "boundary": "stage_b_midi_to_solo_training_resource_probe",
            "midi_to_solo_training_resource_ready": ready,
            "midi_to_solo_mvp_claimed": final_claim,
            "conditioned_generation_completed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloConditionedGenerationProbeTest(unittest.TestCase):
    def test_exports_ranked_conditioned_midi_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_conditioned_generation_report(
                context_report=context_report(),
                resource_probe=resource_probe(),
                output_dir=Path(tmp),
                issue_number=487,
                candidate_count=4,
                export_top_midi_count=3,
                seed_start=487,
                bpm=120,
                bars=8,
                density="medium",
                energy="mid",
                min_note_count=24,
                min_unique_pitch_count=8,
                max_simultaneous_notes=1,
            )
            summary = validate_conditioned_generation_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_exported_candidates=True,
                require_no_final_claim=True,
                min_exported_candidates=3,
                min_note_count=24,
                min_unique_pitch_count=8,
                max_simultaneous_notes=1,
            )

            self.assertTrue(summary["conditioned_generation_probe_completed"])
            self.assertTrue(summary["ranked_midi_candidates_exported"])
            self.assertEqual(summary["exported_candidate_count"], 3)
            self.assertGreaterEqual(summary["qualified_candidate_count"], 3)
            self.assertGreaterEqual(summary["best_note_count"], 24)
            self.assertGreaterEqual(summary["best_unique_pitch_count"], 8)
            self.assertLessEqual(summary["best_max_simultaneous_notes"], 1)
            self.assertFalse(summary["midi_to_solo_mvp_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_resource_probe_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloConditionedGenerationProbeError):
                build_conditioned_generation_report(
                    context_report=context_report(),
                    resource_probe=resource_probe(ready=False),
                    output_dir=Path(tmp),
                    issue_number=487,
                    candidate_count=3,
                    export_top_midi_count=3,
                    seed_start=487,
                    bpm=120,
                    bars=8,
                    density="medium",
                    energy="mid",
                    min_note_count=24,
                    min_unique_pitch_count=8,
                    max_simultaneous_notes=1,
                )

    def test_rejects_final_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloConditionedGenerationProbeError):
                build_conditioned_generation_report(
                    context_report=context_report(),
                    resource_probe=resource_probe(final_claim=True),
                    output_dir=Path(tmp),
                    issue_number=487,
                    candidate_count=3,
                    export_top_midi_count=3,
                    seed_start=487,
                    bpm=120,
                    bars=8,
                    density="medium",
                    energy="mid",
                    min_note_count=24,
                    min_unique_pitch_count=8,
                    max_simultaneous_notes=1,
                )

    def test_rejects_too_few_exported_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_conditioned_generation_report(
                context_report=context_report(),
                resource_probe=resource_probe(),
                output_dir=Path(tmp),
                issue_number=487,
                candidate_count=2,
                export_top_midi_count=2,
                seed_start=487,
                bpm=120,
                bars=8,
                density="medium",
                energy="mid",
                min_note_count=24,
                min_unique_pitch_count=8,
                max_simultaneous_notes=1,
            )
            with self.assertRaises(StageBMidiToSoloConditionedGenerationProbeError):
                validate_conditioned_generation_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_exported_candidates=True,
                    require_no_final_claim=True,
                    min_exported_candidates=3,
                    min_note_count=24,
                    min_unique_pitch_count=8,
                    max_simultaneous_notes=1,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_conditioned_generation_probe")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_candidate_audio_render_package")


if __name__ == "__main__":
    unittest.main()
