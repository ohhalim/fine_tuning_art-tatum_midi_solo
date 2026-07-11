from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_phrase_bank_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_phrase_bank_listening_review_input import (
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_audio import BOUNDARY as AUDIO_RENDER_BOUNDARY
from scripts.run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline import (
    BOUNDARY as PHRASE_BANK_BOUNDARY,
)


def input_guard_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "review_item_count": 3,
        },
        "readiness": {
            "listening_review_input_guard_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": INPUT_GUARD_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def candidate(seed: int, dead_air: float, duration_diversity: float = 0.09) -> dict:
    return {
        "sample_seed": seed,
        "metrics": {
            "note_count": 64,
            "unique_pitch_count": 22,
            "max_simultaneous_notes": 1,
            "dead_air_ratio": dead_air,
            "phrase_coverage_ratio": 1.0,
            "note_density": 4.0,
        },
        "collapse": {
            "per_bar_note_counts": {str(index): 8 for index in range(8)},
            "repeated_pitch_ratio": 0.65,
            "max_same_pitch_repeats": 7,
        },
        "rhythm_profile": {
            "duration_diversity_ratio": duration_diversity,
            "ioi_diversity_ratio": 0.09,
        },
        "approach_resolution": {
            "approach_resolution_ratio": 0.35,
        },
        "phrase_contour": {
            "leap_motion_ratio": 0.0,
        },
    }


def phrase_bank_report() -> dict:
    return {
        "boundary": PHRASE_BANK_BOUNDARY,
        "summary": {
            "exported_candidate_count": 3,
        },
        "objective_gate": {
            "max_dead_air_ratio": 0.65,
        },
        "top_candidates": [
            candidate(635, 0.5873),
            candidate(632, 0.5873),
            candidate(638, 0.6032),
        ],
        "readiness": {
            "phrase_bank_retrieval_baseline_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def audio_render_report() -> dict:
    return {
        "source_boundary": PHRASE_BANK_BOUNDARY,
        "audio_render_boundary": {
            "boundary": AUDIO_RENDER_BOUNDARY,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 3,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "rendered_audio_files": [
            {
                "rank": index,
                "sample_seed": seed,
                "mode": "data_motif_rhythm_phrase_variation",
                "source_midi_path": f"midi/rank_{index}.mid",
                "wav_file": {"path": f"audio/rank_{index}.wav", "duration_seconds": 18.9},
            }
            for index, seed in enumerate([635, 632, 638], start=1)
        ],
    }


class StageBMidiToSoloPhraseBankObjectiveNextTest(unittest.TestCase):
    def test_routes_to_repair_when_objective_risks_remain(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard_report(),
            phrase_bank_report=phrase_bank_report(),
            audio_render_report=audio_render_report(),
            output_dir="out",
            issue_number=640,
            dead_air_review_max=0.45,
            min_rhythm_diversity=0.12,
            min_approach_resolution=0.40,
            max_pitch_reuse_ratio=0.60,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_objective_decision=True,
            require_repair_required=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["objective_keep_candidate_count"], 0)
        self.assertEqual(summary["repair_required_candidate_count"], 3)
        self.assertTrue(summary["all_candidates_require_repair"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPhraseBankObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard_report(quality_claim=True),
                phrase_bank_report=phrase_bank_report(),
                audio_render_report=audio_render_report(),
                output_dir="out",
                issue_number=640,
                dead_air_review_max=0.45,
                min_rhythm_diversity=0.12,
                min_approach_resolution=0.40,
                max_pitch_reuse_ratio=0.60,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_objective_only_next_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe")


if __name__ == "__main__":
    unittest.main()
