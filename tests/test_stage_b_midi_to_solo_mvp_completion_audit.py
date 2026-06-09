from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_mvp_completion import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloMvpCompletionAuditError,
    build_mvp_completion_audit_report,
    validate_mvp_completion_audit_report,
)
from scripts.consolidate_stage_b_midi_to_solo_mvp_current_evidence import (
    BOUNDARY as CURRENT_EVIDENCE_BOUNDARY,
)


def current_evidence(
    *,
    quality_claim: bool = False,
    strict_count: int = 9,
    pitch_contour_supported: bool = True,
) -> dict:
    return {
        "boundary": CURRENT_EVIDENCE_BOUNDARY,
        "readiness": {
            "mvp_current_evidence_consolidated": True,
            "input_contract_ready": True,
            "context_extraction_ready": True,
            "training_resource_ready": True,
            "ranked_midi_candidates_exported": True,
            "technical_wav_path_ready": True,
            "selected_scale_objective_path_complete": True,
            "phrase_bank_cli_technical_path_ready": True,
            "model_conditioned_pitch_contour_objective_path_ready": pitch_contour_supported,
            "current_mvp_technical_execution_evidence_supported": True,
            "current_mvp_objective_repair_evidence_supported": True,
            "midi_to_solo_mvp_current_evidence_supported": pitch_contour_supported,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "ranked_midi_generation": {
            "generation_source": "context_conditioned_fallback",
            "exported_candidate_count": 3,
            "exported_qualified_candidate_count": 3,
        },
        "technical_audio_render": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "duration_min_seconds": 18.617,
            "duration_max_seconds": 18.991,
        },
        "selected_scale_objective_path": {
            "objective_path_supported": True,
            "sample_count": 9,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "dead_air_failure_count": 0,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.2176,
            "target_avg_postprocess_removal_ratio": 0.3,
        },
        "phrase_bank_cli_technical_path": {
            "technical_midi_to_solo_cli_path_ready": True,
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "input_context_bars": 228,
            "preference_fill_allowed": False,
        },
        "model_conditioned_pitch_contour_objective_path": {
            "boundary": (
                "stage_b_midi_to_solo_model_conditioned_input_path_"
                "dead_air_timing_repair_pitch_contour_objective_only_next_decision"
            ),
            "objective_next_decision_completed": True,
            "current_evidence_consolidation_ready": pitch_contour_supported,
            "review_item_count": 3,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 3,
            "max_repaired_interval": 11 if pitch_contour_supported else 13,
            "max_interval_threshold": 12,
            "pitch_contour_target_supported": pitch_contour_supported,
            "max_pitch_changed_ratio": 0.7174,
            "pitch_changed_ratio_review_required": True,
            "audio_review_required": True,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


def readme_text(*, missing_boundary: bool = False) -> str:
    boundary = "stale_boundary" if missing_boundary else CURRENT_EVIDENCE_BOUNDARY
    return "\n".join(
        [
            "- latest evidence boundary: `stage_b_midi_to_solo_mvp_completion_audit`",
            f"- current evidence boundary: `{boundary}`",
            "- current MVP evidence support: `true`",
            "- input MIDI -> context -> ranked MIDI -> WAV technical path: `true`",
            "- selected-scale objective repair path complete: `true`",
            "- phrase-bank CLI technical path included in current evidence: `true`",
            "- model-conditioned pitch-contour objective path ready: `true`",
            "- model-conditioned pitch-contour changed-ratio review required: `true`",
            "- README evidence refreshed: `true`",
            "- human/audio preference claim: `false`",
            "- MIDI-to-solo musical quality claim: `false`",
            "- broad trained-model quality claim: `false`",
            "- Brad style adaptation claim: `false`",
        ]
    )


class StageBMidiToSoloMvpCompletionAuditTest(unittest.TestCase):
    def test_audits_technical_mvp_completion_without_quality_claim(self) -> None:
        report = build_mvp_completion_audit_report(
            current_evidence=current_evidence(),
            readme_text=readme_text(),
            output_dir=Path("outputs/audit"),
            issue_number=616,
        )
        summary = validate_mvp_completion_audit_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_technical_mvp_completion=True,
            require_no_quality_claim=True,
            require_model_conditioned_pitch_contour_objective=True,
        )

        self.assertTrue(summary["technical_model_core_mvp_completed"])
        self.assertTrue(summary["input_to_ranked_midi_completed"])
        self.assertTrue(summary["input_to_rendered_wav_completed"])
        self.assertTrue(summary["selected_scale_objective_repair_completed"])
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertFalse(summary["musical_quality_mvp_completed"])
        self.assertFalse(summary["human_audio_preference_completed"])
        self.assertFalse(summary["product_mvp_completed"])
        self.assertEqual(summary["generation_source"], "context_conditioned_fallback")
        self.assertEqual(summary["exported_candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["objective_sample_count"], 9)
        self.assertEqual(summary["objective_strict_valid_sample_count"], 9)
        self.assertEqual(summary["objective_dead_air_failure_count"], 0)
        self.assertTrue(summary["phrase_bank_cli_technical_path_ready"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertTrue(summary["model_conditioned_pitch_contour_objective_completed"])
        self.assertTrue(summary["model_conditioned_pitch_contour_objective_path_ready"])
        self.assertEqual(summary["model_conditioned_pitch_contour_max_interval"], 11)
        self.assertEqual(summary["model_conditioned_pitch_contour_max_interval_threshold"], 12)
        self.assertTrue(summary["model_conditioned_pitch_contour_target_supported"])
        self.assertTrue(
            summary["model_conditioned_pitch_contour_pitch_changed_ratio_review_required"]
        )
        self.assertTrue(summary["model_conditioned_pitch_contour_audio_review_required"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_missing_readme_evidence_boundary(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(),
                readme_text=readme_text(missing_boundary=True),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(quality_claim=True),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_objective_strict_shortfall(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(strict_count=8),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_missing_pitch_contour_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(pitch_contour_supported=False),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=712,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_completion_audit")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_quality_gap_decision")


if __name__ == "__main__":
    unittest.main()
