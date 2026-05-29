from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.summarize_stage_b_duration_coverage_fill_broader_repeatability_sweep import (
    StageBDurationCoverageBroaderRepeatabilitySweepError,
    build_broader_repeatability_report,
    validate_broader_repeatability_report,
)


def write_source_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="repeatability_source")
    rows = [
        (70, 0.120967, 0.483870),
        (77, 0.725805, 0.846772),
        (75, 0.967740, 1.088708),
        (79, 1.209675, 1.572578),
        (72, 1.572578, 1.693545),
        (65, 1.935480, 2.056448),
        (68, 2.298382, 2.419350),
        (67, 2.419350, 2.540318),
        (63, 2.903220, 3.024188),
        (70, 3.024188, 3.145155),
        (72, 3.266122, 3.387090),
        (65, 3.387090, 3.508058),
    ]
    for pitch, start, end in rows:
        piano.notes.append(pretty_midi.Note(velocity=84, pitch=pitch, start=start, end=end))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_source_run(root: Path, run_id: str, sample_index: int) -> None:
    source_root = root / run_id
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "report.json").write_text(
        json.dumps(
            {
                "request": {
                    "bpm": 124,
                    "chord_progression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
                    "bars": 2,
                    "density": "medium",
                    "energy": "mid",
                    "temperature": 0.82,
                    "top_k": 7,
                    "top_p": None,
                    "seed": 109,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_source_midi(source_root / "samples" / f"stage_b_sample_{sample_index}.mid")


def next_decision(*, critical: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_next_decision_v1",
        "decision": {
            "next_boundary": "broader_repeatability_sweep",
            "auto_progress_allowed": True,
            "critical_user_input_required": critical,
        },
    }


def user_listening_consolidation(*, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_consolidation_v1",
        "candidate_id": "current_duration_fill_keep",
        "consolidated_claim_boundary": {
            "single_user_human_audio_preference_claimed": True,
            "broad_model_quality_claimed": broad_claim,
        },
    }


def current_duration_fill_summary() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair_v1",
        "variant_count": 4,
        "qualified_variant_count": 2,
        "source_candidate": {"candidate_id": "current_source"},
        "selected_candidate": {"candidate_id": "current_duration_fill_keep"},
        "repair_summary": {
            "selected_candidate_id": "current_duration_fill_keep",
            "selected_fill_addition_count": 6,
            "qualified": True,
            "duration_coverage_fill_improved": True,
            "baseline_dead_air_ratio": 0.5714285714285714,
            "selected_dead_air_ratio": 0.29411764705882354,
            "dead_air_delta_from_baseline": 0.277311,
            "selected_focused_note_count": 18,
            "selected_focused_unique_pitch_count": 15,
            "selected_adjacent_pitch_repeats": 0,
            "selected_max_interval": 7,
            "claim_boundary": "postprocess_duration_coverage_fill_candidate",
        },
    }


def distinct_sample_seed_sweep() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep_v1",
        "top_candidates": [
            {
                "candidate_id": "source_a",
                "source_run_id": "run_a",
                "source_seed": 109,
                "sample_index": 1,
                "sample_seed": 155,
                "repair_rank": 1,
                "qualified": True,
                "note_count": 17,
                "unique_pitch_count": 6,
                "dead_air_ratio": 0.1,
                "focused_note_count": 13,
                "focused_unique_pitch_count": 6,
                "focused_adjacent_pitch_repeats": 1,
                "focused_max_interval": 3,
            },
            {
                "candidate_id": "source_b",
                "source_run_id": "run_b",
                "source_seed": 109,
                "sample_index": 2,
                "sample_seed": 131,
                "repair_rank": 2,
                "qualified": True,
                "note_count": 17,
                "unique_pitch_count": 7,
                "dead_air_ratio": 0.5714285714285714,
                "focused_note_count": 12,
                "focused_unique_pitch_count": 7,
                "focused_adjacent_pitch_repeats": 1,
                "focused_max_interval": 11,
            },
        ],
    }


class StageBDurationCoverageFillBroaderRepeatabilitySweepTest(unittest.TestCase):
    def test_records_partial_dead_air_repeatability_boundary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generation_root = root / "outputs" / "stage_b_generation_probe"
            write_source_run(generation_root, "run_a", 1)
            write_source_run(generation_root, "run_b", 2)

            report = build_broader_repeatability_report(
                next_decision=next_decision(),
                user_listening_consolidation=user_listening_consolidation(),
                duration_fill_summary=current_duration_fill_summary(),
                distinct_sample_seed_sweep=distinct_sample_seed_sweep(),
                output_dir=root / "repeatability",
                generation_output_root=generation_root,
                max_source_candidates=2,
                min_source_candidates=2,
                min_qualified_source_candidates=2,
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
            summary = validate_broader_repeatability_report(
                report,
                expected_boundary="qualified_gate_repeatability_with_partial_dead_air_gain",
                min_source_candidates=2,
                min_qualified_source_candidates=2,
                require_no_broad_quality_claim=True,
            )

            self.assertEqual(summary["source_candidate_count"], 2)
            self.assertEqual(summary["qualified_source_candidate_count"], 2)
            self.assertEqual(summary["dead_air_improved_source_candidate_count"], 1)
            self.assertEqual(summary["total_variant_count"], 8)
            self.assertFalse(summary["broad_model_quality_claimed"])
            self.assertIn("uniform_dead_air_gain_across_distinct_sources", report["not_proven"])

    def test_rejects_critical_next_decision(self) -> None:
        with self.assertRaises(StageBDurationCoverageBroaderRepeatabilitySweepError):
            build_broader_repeatability_report(
                next_decision=next_decision(critical=True),
                user_listening_consolidation=user_listening_consolidation(),
                duration_fill_summary=current_duration_fill_summary(),
                distinct_sample_seed_sweep=distinct_sample_seed_sweep(),
                output_dir=Path("outputs/repeatability"),
                generation_output_root=Path("missing"),
                max_source_candidates=2,
                min_source_candidates=2,
                min_qualified_source_candidates=2,
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

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageBroaderRepeatabilitySweepError):
            build_broader_repeatability_report(
                next_decision=next_decision(),
                user_listening_consolidation=user_listening_consolidation(broad_claim=True),
                duration_fill_summary=current_duration_fill_summary(),
                distinct_sample_seed_sweep=distinct_sample_seed_sweep(),
                output_dir=Path("outputs/repeatability"),
                generation_output_root=Path("missing"),
                max_source_candidates=2,
                min_source_candidates=2,
                min_qualified_source_candidates=2,
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
