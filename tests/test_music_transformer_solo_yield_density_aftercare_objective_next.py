from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_music_transformer_solo_yield_density_aftercare_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY_DENSITY_AFTERCARE,
    NEXT_BOUNDARY_INTERVAL_CONTOUR,
    NEXT_BOUNDARY_LISTENING_REVIEW_FILL,
    SCHEMA_VERSION,
    build_decision,
    validate_report,
)
from scripts.guard_music_transformer_solo_yield_density_aftercare_listening_input import (
    SCHEMA_VERSION as INPUT_GUARD_SCHEMA_VERSION,
)
from scripts.run_music_transformer_solo_yield_density_aftercare_sweep import (
    SCHEMA_VERSION as REPAIR_SWEEP_SCHEMA_VERSION,
)


def after_profile(
    *,
    chord_tone_ratio: float = 0.55,
    note_count: int = 32,
    direction_change_ratio: float = 0.56,
    max_gap: float = 0.55,
    max_interval: int = 7,
) -> dict:
    return {
        "midi_note_count": note_count,
        "midi_unique_pitch_count": 12,
        "midi_pitch_span": 14,
        "midi_max_abs_interval": max_interval,
        "midi_direction_change_ratio": direction_change_ratio,
        "midi_max_gap_seconds": max_gap,
        "midi_avg_gap_seconds": 0.25,
        "midi_chord_tone_ratio": chord_tone_ratio,
        "midi_tension_ratio": 1.0 - chord_tone_ratio,
        "final_landing_chord": "Ebmaj7",
        "final_landing_pitch": 63,
        "final_landing_is_chord_tone": True,
    }


def repair_sweep(rows: list[dict]) -> dict:
    return {
        "schema_version": REPAIR_SWEEP_SCHEMA_VERSION,
        "output_dir": "outputs/repair",
        "candidate_repairs": [
            {
                "review_index": index + 1,
                "case_label": f"case_{index + 1}",
                "repaired_midi_path": f"outputs/repair/candidate_{index + 1}.mid",
                "source_wav_path": f"outputs/repair/candidate_{index + 1}.wav",
                "after_profile": row,
            }
            for index, row in enumerate(rows)
        ],
        "aggregate": {
            "candidate_count": len(rows),
            "target_supported": True,
            "low_note_count_after": 0,
            "chord_tone_ratio_decrease_count": 0,
            "weak_direction_change_count_after": 0,
            "final_landing_not_chord_tone_count_after": 0,
            "wide_interval_review_count_after": 0,
        },
        "readiness": {
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def guard_report(*, preference_fill_allowed: bool = False) -> dict:
    return {
        "schema_version": INPUT_GUARD_SCHEMA_VERSION,
        "output_dir": "outputs/guard",
        "input_validation": {
            "validated_listening_input_present": preference_fill_allowed,
            "pending_candidate_field_count": 0 if preference_fill_allowed else 12,
        },
        "readiness": {
            "preference_fill_allowed": preference_fill_allowed,
            "objective_only_next_decision_required": not preference_fill_allowed,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldDensityAftercareObjectiveNextTest(unittest.TestCase):
    def test_selects_density_aftercare_when_low_note_count_is_largest_residual(self) -> None:
        rows = [
            after_profile(note_count=29),
            after_profile(note_count=28),
            after_profile(max_interval=8),
            after_profile(),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(repair_sweep(rows), guard_report(), output_dir=Path(temp_dir))
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 4)
        self.assertEqual(summary["midi_low_chord_tone_ratio_count"], 0)
        self.assertEqual(summary["low_note_count_for_4bar_count"], 2)
        self.assertEqual(summary["wide_interval_review_count"], 1)
        self.assertEqual(summary["selected_next_target"], "density_aftercare")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY_DENSITY_AFTERCARE)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_selects_interval_contour_when_wide_interval_is_only_residual(self) -> None:
        rows = [
            after_profile(max_interval=8),
            after_profile(max_interval=8),
            after_profile(max_interval=7),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(repair_sweep(rows), guard_report(), output_dir=Path(temp_dir))
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["wide_interval_review_count"], 2)
        self.assertEqual(summary["selected_next_target"], "interval_contour_aftercare")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY_INTERVAL_CONTOUR)

    def test_validated_input_routes_to_listening_review_fill(self) -> None:
        rows = [
            after_profile(note_count=29),
            after_profile(max_interval=8),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision(
                repair_sweep(rows),
                guard_report(preference_fill_allowed=True),
                output_dir=Path(temp_dir),
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertTrue(summary["validated_listening_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])
        self.assertEqual(summary["selected_next_target"], "listening_review_fill")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY_LISTENING_REVIEW_FILL)


if __name__ == "__main__":
    unittest.main()
