from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankRetrievalBaselineError,
    build_phrase_bank_retrieval_report,
    validate_phrase_bank_retrieval_report,
)


def context_report() -> dict:
    bars = [
        ("C", "maj7", 36, "pitch_class_inference", 0.9),
        ("F", "dom7", 41, "pitch_class_inference", 0.9),
        ("G", "dom7", 43, "pitch_class_inference", 0.9),
        ("C", "maj7", 36, "pitch_class_inference", 0.9),
        ("C", "maj7", None, "carry_forward_empty_bar", 0.45),
        ("F", "dom7", None, "carry_forward_empty_bar", 0.45),
        ("G", "dom7", None, "carry_forward_empty_bar", 0.45),
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


def template_report() -> dict:
    rhythm_key = {
        "position_deltas": [0, 2, 5, 7],
        "duration_steps": [2, 2, 3, 2],
    }
    contour_key = {
        "pitch_intervals": [0, 2, 4, 5],
        "melodic_intervals": [2, 2, 1],
    }
    full_key = {
        "position_deltas": [0, 2, 5, 7],
        "duration_steps": [2, 2, 3, 2],
        "pitch_intervals": [0, 2, 4, 5],
    }
    return {
        "summary": {
            "source_record_count": 2,
            "motif_count": 16,
            "unique_rhythm_template_count": 2,
            "unique_contour_template_count": 2,
            "unique_full_template_count": 2,
            "top_rhythm_template_support_ratio": 0.25,
            "top_contour_template_support_ratio": 0.25,
            "top_full_template_support_ratio": 0.25,
            "top_rhythm_templates": [
                {"rank": 1, "count": 4, "support_ratio": 0.25, "key": rhythm_key, "examples": []},
                {
                    "rank": 2,
                    "count": 3,
                    "support_ratio": 0.1875,
                    "key": {"position_deltas": [0, 3, 4, 7], "duration_steps": [3, 1, 2, 2]},
                    "examples": [],
                },
            ],
            "top_contour_templates": [
                {"rank": 1, "count": 4, "support_ratio": 0.25, "key": contour_key, "examples": []},
                {
                    "rank": 2,
                    "count": 3,
                    "support_ratio": 0.1875,
                    "key": {"pitch_intervals": [0, -1, 2, 4], "melodic_intervals": [-1, 3, 2]},
                    "examples": [],
                },
            ],
            "top_full_templates": [
                {"rank": 1, "count": 4, "support_ratio": 0.25, "key": full_key, "examples": []},
                {
                    "rank": 2,
                    "count": 3,
                    "support_ratio": 0.1875,
                    "key": {
                        "position_deltas": [0, 3, 4, 7],
                        "duration_steps": [3, 1, 2, 2],
                        "pitch_intervals": [0, -1, 2, 4],
                    },
                    "examples": [],
                },
            ],
        }
    }


class StageBMidiToSoloPhraseBankRetrievalBaselineTest(unittest.TestCase):
    def test_exports_phrase_bank_midi_candidates_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_phrase_bank_retrieval_report(
                context_report=context_report(),
                output_dir=Path(tmp),
                run_id="test_phrase_bank",
                issue_number=632,
                template_report=template_report(),
                input_dir=Path("midi_dataset/midi/studio"),
                modes=[
                    "data_motif_rhythm_phrase_variation",
                    "data_motif_contour_landing_repair",
                    "data_motif_phrase_recovery",
                ],
                candidate_count=4,
                export_top_midi_count=2,
                seed_start=632,
                bpm=120,
                bars=8,
                density="medium",
                energy="mid",
                note_groups_per_bar=8,
                max_sequence=384,
                max_simultaneous_notes=1,
                min_note_count=24,
                min_unique_pitch_count=8,
                min_phrase_coverage_ratio=0.75,
                max_dead_air_ratio=0.65,
                max_files=1,
                window_bars=8,
                window_stride_bars=4,
                min_window_target_notes=16,
                motif_length=4,
                max_bar_span=2,
                max_records=8,
                template_top_n=8,
            )
            summary = validate_phrase_bank_retrieval_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_exported_candidates=True,
                require_no_final_claim=True,
                min_exported_candidates=2,
                min_note_count=24,
                min_unique_pitch_count=8,
                max_simultaneous_notes=1,
            )

            self.assertTrue(summary["phrase_bank_template_extracted"])
            self.assertTrue(summary["phrase_bank_retrieval_baseline_completed"])
            self.assertTrue(summary["ranked_midi_candidates_exported"])
            self.assertEqual(summary["exported_candidate_count"], 2)
            self.assertGreaterEqual(summary["motif_count"], 1)
            self.assertFalse(summary["midi_to_solo_mvp_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_missing_template_evidence(self) -> None:
        broken = {"summary": {"source_record_count": 0, "motif_count": 0}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloPhraseBankRetrievalBaselineError):
                build_phrase_bank_retrieval_report(
                    context_report=context_report(),
                    output_dir=Path(tmp),
                    run_id="test_phrase_bank",
                    issue_number=632,
                    template_report=broken,
                    input_dir=Path("midi_dataset/midi/studio"),
                    modes=["data_motif_phrase_recovery"],
                    candidate_count=1,
                    export_top_midi_count=1,
                    seed_start=632,
                    bpm=120,
                    bars=8,
                    density="medium",
                    energy="mid",
                    note_groups_per_bar=8,
                    max_sequence=384,
                    max_simultaneous_notes=1,
                    min_note_count=24,
                    min_unique_pitch_count=8,
                    min_phrase_coverage_ratio=0.75,
                    max_dead_air_ratio=0.65,
                    max_files=1,
                    window_bars=8,
                    window_stride_bars=4,
                    min_window_target_notes=16,
                    motif_length=4,
                    max_bar_span=2,
                    max_records=8,
                    template_top_n=8,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_retrieval_baseline")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_audio_render_package")


if __name__ == "__main__":
    unittest.main()
