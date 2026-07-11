from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_objective_next import (
    BOUNDARY,
    CURRENT_EVIDENCE_NEXT_BOUNDARY,
    REPAIR_NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input import (
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_audio import (
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
)


SOURCE_EVIDENCE = {
    "phrase_bank_cli_technical_path_completed": True,
    "cli_candidate_count": 3,
    "cli_rendered_audio_file_count": 3,
    "cli_input_context_bars": 228,
    "cli_preference_fill_allowed": False,
}


def input_guard(*, preference_fill_allowed: bool = False, quality_claim: bool = False) -> dict:
    return {
        "boundary": INPUT_GUARD_BOUNDARY,
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": preference_fill_allowed,
            "review_item_count": 3,
            "required_input_field_count": 4,
            "source_evidence": dict(SOURCE_EVIDENCE),
        },
        "readiness": {
            "listening_review_input_guard_completed": True,
            "human_audio_preference_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": INPUT_GUARD_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def candidate_export(*, dead_air: float = 0.6522) -> dict:
    rows = [
        {
            "rank": index,
            "sample_index": index,
            "sample_seed": 496 + index,
            "export_midi_path": f"midi/rank_{index:02d}.mid",
            "note_count": 24,
            "unique_pitch_count": 20,
            "max_simultaneous_notes": 1,
            "chord_tone_ratio": 0.6,
            "dead_air_ratio": dead_air,
            "phrase_coverage_ratio": 0.96,
            "position_span_ratio": 0.94,
            "postprocess_removal_ratio": 0.0,
        }
        for index in range(1, 4)
    ]
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "probe_source": dict(SOURCE_EVIDENCE),
        "top_candidates": rows,
        "summary": {
            "exported_candidate_count": 3,
            "best_note_count": 24,
            "best_unique_pitch_count": 20,
            "best_dead_air_ratio": dead_air,
        },
        "readiness": {
            "model_conditioned_input_path_candidate_export_completed": True,
            "ranked_midi_candidates_exported": True,
            "model_conditioned_ranked_input_path_contract_matched": True,
            "fallback_replacement_candidate_export_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
    }


def audio_render(*, dead_air: float = 0.6522) -> dict:
    return {
        "source_boundary": CANDIDATE_EXPORT_BOUNDARY,
        "candidate_export_source": dict(SOURCE_EVIDENCE),
        "audio_render_boundary": {
            "boundary": AUDIO_RENDER_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "model_conditioned_ranked_audio_render_completed": True,
            "fallback_replacement_candidate_export_ready": True,
            "fallback_replacement_technical_path_ready": True,
            "fallback_replacement_ready": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
        },
        "rendered_audio_files": [
            {
                "rank": index,
                "sample_index": index,
                "sample_seed": 496 + index,
                "source_midi_path": f"midi/rank_{index:02d}.mid",
                "source_note_count": 24,
                "source_unique_pitch_count": 20,
                "source_chord_tone_ratio": 0.6,
                "source_dead_air_ratio": dead_air,
                "wav_file": {
                    "path": f"audio/rank_{index:02d}.wav",
                    "duration_seconds": 20.0,
                    "sample_rate": 44100,
                    "sha256": f"hash{index}",
                },
            }
            for index in range(1, 4)
        ],
    }


class StageBMidiToSoloModelConditionedInputPathObjectiveNextTest(unittest.TestCase):
    def test_routes_to_repair_when_dead_air_threshold_fails(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard(),
            candidate_export_report=candidate_export(),
            audio_render_report=audio_render(),
            output_dir="out",
            issue_number=686,
            expected_count=3,
            dead_air_threshold=0.5,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=REPAIR_NEXT_BOUNDARY,
            require_objective_decision=True,
            require_repair_required=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["model_conditioned_technical_path_ready"])
        self.assertEqual(summary["dead_air_failure_count"], 3)
        self.assertTrue(summary["all_candidates_dead_air_failure"])
        self.assertTrue(summary["dead_air_timing_repair_required"])
        self.assertFalse(summary["current_evidence_consolidation_ready"])
        self.assertFalse(summary["preference_fill_allowed"])

    def test_routes_to_current_evidence_when_dead_air_threshold_passes(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard(),
            candidate_export_report=candidate_export(dead_air=0.2),
            audio_render_report=audio_render(dead_air=0.2),
            output_dir="out",
            issue_number=686,
            expected_count=3,
            dead_air_threshold=0.5,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=CURRENT_EVIDENCE_NEXT_BOUNDARY,
            require_objective_decision=True,
            require_repair_required=False,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["dead_air_timing_repair_required"])
        self.assertTrue(summary["current_evidence_consolidation_ready"])

    def test_rejects_preference_fill_allowed(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(preference_fill_allowed=True),
                candidate_export_report=candidate_export(),
                audio_render_report=audio_render(),
                output_dir="out",
                issue_number=686,
                expected_count=3,
                dead_air_threshold=0.5,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloModelConditionedInputPathObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(quality_claim=True),
                candidate_export_report=candidate_export(),
                audio_render_report=audio_render(),
                output_dir="out",
                issue_number=686,
                expected_count=3,
                dead_air_threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
