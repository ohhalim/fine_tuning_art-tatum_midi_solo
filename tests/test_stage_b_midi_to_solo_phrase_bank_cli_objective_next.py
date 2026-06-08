from __future__ import annotations

import unittest

from scripts.check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke import (
    BOUNDARY as USER_INPUT_BOUNDARY,
    NEXT_BOUNDARY as USER_INPUT_NEXT_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_phrase_bank_cli_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankCliObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input import (
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke import (
    BOUNDARY as AUDIO_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_NEXT_BOUNDARY,
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
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": INPUT_GUARD_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def user_input_smoke_report() -> dict:
    candidates = []
    for rank, seed, dead_air in [(1, 635, 0.1895), (2, 632, 0.2105), (3, 638, 0.2211)]:
        candidates.append(
            {
                "rank": rank,
                "sample_seed": seed,
                "repaired_midi_path": f"midi/rank_{rank}.mid",
                "objective_supported": True,
                "note_count": 96,
                "unique_pitch_count": 22,
                "max_simultaneous_notes": 1,
                "dead_air_ratio": dead_air,
                "phrase_coverage_ratio": 1.0,
            }
        )
    return {
        "boundary": USER_INPUT_BOUNDARY,
        "input": {
            "midi_path": "midi_dataset/midi/studio/Geri Allen/Home Grown/Alone Together.midi",
            "explicit_input_used": True,
        },
        "objective_summary": {
            "candidate_count": 3,
            "objective_supported_candidate_count": 3,
            "all_candidates_objective_supported": True,
            "repaired_midi_file_count": 3,
            "input_context_bars": 228,
            "min_dead_air_ratio": 0.1895,
            "max_dead_air_ratio": 0.2211,
        },
        "candidate_manifest": candidates,
        "readiness": {
            "user_input_smoke_completed": True,
            "explicit_input_path_used": True,
            "ranked_repaired_midi_exported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": USER_INPUT_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def audio_render_report() -> dict:
    rows = []
    for rank, seed in [(1, 635), (2, 632), (3, 638)]:
        rows.append(
            {
                "rank": rank,
                "sample_seed": seed,
                "wav_file": {
                    "path": f"audio/rank_{rank}.wav",
                    "duration_seconds": 18.9,
                    "sample_rate": 44100,
                    "sha256": f"{seed}" * 16,
                },
            }
        )
    return {
        "source_boundary": USER_INPUT_BOUNDARY,
        "audio_render_boundary": {
            "boundary": AUDIO_BOUNDARY,
            "technical_wav_validation": True,
            "cli_user_input_audio_render_completed": True,
            "rendered_audio_file_count": 3,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "rendered_audio_files": rows,
        "decision": {
            "next_boundary": AUDIO_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseBankCliObjectiveNextTest(unittest.TestCase):
    def test_routes_technical_cli_path_to_current_evidence_consolidation(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard_report(),
            user_input_smoke_report=user_input_smoke_report(),
            audio_render_report=audio_render_report(),
            output_dir="out",
            issue_number=662,
            min_candidate_count=3,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_objective_decision=True,
            require_current_evidence_ready=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["technical_midi_to_solo_cli_path_ready"])
        self.assertTrue(summary["mvp_current_evidence_consolidation_ready"])
        self.assertTrue(summary["explicit_input_used"])
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["objective_supported_candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPhraseBankCliObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard_report(quality_claim=True),
                user_input_smoke_report=user_input_smoke_report(),
                audio_render_report=audio_render_report(),
                output_dir="out",
                issue_number=662,
                min_candidate_count=3,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_mvp_current_evidence_consolidation")


if __name__ == "__main__":
    unittest.main()
