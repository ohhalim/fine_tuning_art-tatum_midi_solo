from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BOUNDARY as FINAL_STATUS_BOUNDARY,
    NEXT_BOUNDARY as FINAL_STATUS_NEXT_BOUNDARY,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloPostMvpQualityIterationPlanError,
    build_post_mvp_quality_iteration_plan_report,
    validate_post_mvp_quality_iteration_plan_report,
)


def final_status_audit(
    *,
    quality_claim: bool = False,
    listening_gap_open: bool = True,
    cli_count: int = 3,
    wav_count: int = 3,
    outside_ready: bool = True,
    outside_wav_count: int = 6,
    outside_risk_after: int = 0,
) -> dict:
    return {
        "boundary": FINAL_STATUS_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_mvp_delivery_package",
        "final_status": {
            "technical_mvp_complete": True,
            "technical_mvp_ready_for_local_review": True,
            "readme_final_evidence_reflected": True,
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
            "changed_ratio_repair_audio_evidence_ready": True,
            "cli_candidate_count": cli_count,
            "changed_ratio_repair_wav_count": wav_count,
            "outside_soloing_repair_evidence_ready": outside_ready,
            "outside_soloing_repair_wav_count": outside_wav_count,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_pitch_role_risk_count_after": outside_risk_after,
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            "listening_review_quality_gap_open": listening_gap_open,
            "raw_artifact_upload_required": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
        },
        "readiness": {
            "final_status_audit_completed": True,
            "technical_mvp_complete": True,
            "technical_mvp_ready_for_local_review": True,
            "readme_final_evidence_reflected": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": FINAL_STATUS_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "next_recommended_issue": "Stage B MIDI-to-solo post-MVP musical quality iteration plan",
    }


class StageBMidiToSoloPostMvpQualityIterationPlanTest(unittest.TestCase):
    def test_selects_quality_rubric_baseline_without_quality_claim(self) -> None:
        report = build_post_mvp_quality_iteration_plan_report(
            final_status_audit=final_status_audit(),
            output_dir=Path("outputs/post_mvp_quality"),
            issue_number=744,
        )
        summary = validate_post_mvp_quality_iteration_plan_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            require_quality_rubric=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["post_mvp_quality_iteration_plan_completed"])
        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["local_review_ready"])
        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertTrue(summary["quality_rubric_required"])
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        self.assertTrue(summary["candidate_failure_labeling_required"])
        self.assertTrue(summary["targeted_quality_repair_sweep_required"])
        self.assertTrue(summary["audio_review_package_required"])
        self.assertGreaterEqual(summary["ordered_work_count"], 4)
        self.assertGreaterEqual(summary["quality_failure_taxonomy_seed_count"], 7)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo quality rubric baseline",
        )

    def test_rejects_wrong_source_boundary(self) -> None:
        source = final_status_audit()
        source["boundary"] = "wrong_boundary"
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=source,
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(quality_claim=True),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )

    def test_rejects_closed_listening_gap(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(listening_gap_open=False),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )

    def test_rejects_low_candidate_or_wav_count(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(cli_count=2),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(wav_count=2),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )

    def test_rejects_incomplete_outside_soloing_repair_evidence(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_ready=False),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_wav_count=5),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_risk_after=1),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=744,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_post_mvp_quality_iteration_plan")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_quality_rubric_baseline")
        self.assertEqual(SELECTED_TARGET, "quality_rubric_baseline")


if __name__ == "__main__":
    unittest.main()
