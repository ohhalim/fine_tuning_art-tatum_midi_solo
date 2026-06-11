from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_music_transformer_solo_yield_post_final_quality_iteration import (
    SoloYieldPostFinalQualityIterationDecisionError,
    build_decision_report,
    validate_decision_report,
)


def final_status_audit(
    *,
    technical_ready: bool = True,
    reproducible_handoff: bool = True,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_final_status_audit_v1",
        "output_dir": "outputs/final_status_audit",
        "final_status": {
            "technical_mvp_evidence_ready": technical_ready,
            "strict_valid_sample_count": 40,
            "sample_count": 40,
            "grammar_gate_sample_count": 40,
            "final_handoff_midi_count": 8,
            "final_handoff_wav_count": 8,
            "reproducible_handoff": reproducible_handoff,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "readiness": {
            "final_status_audit_completed": True,
            "technical_mvp_evidence_ready": technical_ready,
            "human_listening_review_pending": True,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def objective_decision(*, selected_count: int = 4) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_objective_next_decision_v1",
        "output_dir": "outputs/objective_decision",
        "objective_summary": {
            "candidate_count": 8,
            "score_min": 233.1337,
            "score_max": 234.10437,
            "score_avg": 233.61407175,
            "note_count_avg": 29.875,
            "dead_air_min": 0.625,
            "dead_air_max": 0.7241379310344828,
        },
        "readiness": {
            "objective_only_next_decision_completed": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "selected_objective_candidate_count": selected_count,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldPostFinalQualityIterationDecisionTest(unittest.TestCase):
    def test_selects_objective_quality_rubric_when_listening_input_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_decision_report(
                final_status_audit=final_status_audit(),
                objective_decision=objective_decision(),
                output_dir=Path(temp_dir) / "decision",
            )
        summary = validate_decision_report(
            report,
            expected_target="objective_candidate_quality_rubric",
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["technical_mvp_evidence_ready"])
        self.assertEqual(summary["strict_valid_sample_count"], 40)
        self.assertEqual(summary["sample_count"], 40)
        self.assertTrue(summary["reproducible_handoff"])
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertEqual(summary["candidate_count"], 8)
        self.assertEqual(summary["selected_objective_candidate_count"], 4)
        self.assertEqual(summary["selected_next_target"], "objective_candidate_quality_rubric")
        self.assertEqual(
            summary["next_boundary"],
            "music_transformer_solo_yield_objective_quality_rubric_baseline",
        )
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_not_ready_final_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldPostFinalQualityIterationDecisionError):
                build_decision_report(
                    final_status_audit=final_status_audit(technical_ready=False),
                    objective_decision=objective_decision(),
                    output_dir=Path(temp_dir) / "decision",
                )

    def test_rejects_non_reproducible_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldPostFinalQualityIterationDecisionError):
                build_decision_report(
                    final_status_audit=final_status_audit(reproducible_handoff=False),
                    objective_decision=objective_decision(),
                    output_dir=Path(temp_dir) / "decision",
                )

    def test_rejects_missing_selected_objective_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldPostFinalQualityIterationDecisionError):
                build_decision_report(
                    final_status_audit=final_status_audit(),
                    objective_decision=objective_decision(selected_count=0),
                    output_dir=Path(temp_dir) / "decision",
                )

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldPostFinalQualityIterationDecisionError):
                build_decision_report(
                    final_status_audit=final_status_audit(quality_claim=True),
                    objective_decision=objective_decision(),
                    output_dir=Path(temp_dir) / "decision",
                )


if __name__ == "__main__":
    unittest.main()
