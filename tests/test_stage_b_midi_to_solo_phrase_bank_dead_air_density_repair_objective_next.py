from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_next import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input import (
    BOUNDARY as INPUT_GUARD_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as INPUT_GUARD_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio import (
    BOUNDARY as AUDIO_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe import (
    BOUNDARY as REPAIR_BOUNDARY,
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


def repair_probe_report() -> dict:
    rows = []
    for rank, seed, dead_air in [(1, 635, 0.1895), (2, 632, 0.2105), (3, 638, 0.2211)]:
        rows.append(
            {
                "rank": rank,
                "sample_seed": seed,
                "repaired_midi_path": f"midi/rank_{rank}.mid",
                "repaired_metrics": {
                    "note_count": 96,
                    "unique_pitch_count": 20,
                    "max_simultaneous_notes": 1,
                    "dead_air_ratio": dead_air,
                    "phrase_coverage_ratio": 1.0,
                },
                "repair_gate": {"qualified": True, "flags": []},
            }
        )
    return {
        "boundary": REPAIR_BOUNDARY,
        "summary": {
            "repair_probe_target_passed": True,
            "qualified_repaired_candidate_count": 3,
        },
        "repaired_candidates": rows,
        "readiness": {
            "dead_air_density_repair_probe_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


def audio_package_report() -> dict:
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
        "audio_render_boundary": {
            "boundary": AUDIO_BOUNDARY,
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
        "rendered_audio_files": rows,
        "decision": {
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_objective_supported_repaired_evidence_to_cli_package(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard_report(),
            repair_probe_report=repair_probe_report(),
            audio_package_report=audio_package_report(),
            output_dir="out",
            issue_number=650,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_objective_decision=True,
            require_cli_ready=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["objective_supported_candidate_count"], 3)
        self.assertTrue(summary["all_repaired_candidates_objective_supported"])
        self.assertTrue(summary["technical_wav_validation"])
        self.assertTrue(summary["cli_mvp_package_ready"])
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPhraseBankDeadAirDensityRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard_report(quality_claim=True),
                repair_probe_report=repair_probe_report(),
                audio_package_report=audio_package_report(),
                output_dir="out",
                issue_number=650,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision",
        )
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_phrase_bank_cli_mvp_package")


if __name__ == "__main__":
    unittest.main()
