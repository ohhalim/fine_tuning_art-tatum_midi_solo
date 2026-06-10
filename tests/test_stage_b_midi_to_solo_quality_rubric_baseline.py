from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloQualityRubricBaselineError,
    build_quality_rubric_baseline_report,
    validate_quality_rubric_baseline_report,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
)


def post_mvp_quality_plan(
    *,
    selected_target: str = POST_MVP_SELECTED_TARGET,
    quality_claim: bool = False,
    ordered_work_count: int = 4,
    taxonomy_count: int = 7,
    outside_ready: bool = True,
    outside_wav_count: int = 6,
    outside_risk_after: int = 0,
) -> dict:
    ordered_targets = [
        "quality_rubric_baseline",
        "candidate_failure_labeling",
        "targeted_quality_repair_sweep",
        "audio_review_package",
    ][:ordered_work_count]
    return {
        "boundary": POST_MVP_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_final_status_audit",
        "post_mvp_status": {
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "outside_soloing_repair_evidence_ready": outside_ready,
            "outside_soloing_repair_wav_count": outside_wav_count,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_pitch_role_risk_count_after": outside_risk_after,
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
        },
        "selected_next_target": {
            "selected_target": selected_target,
            "selected_next_boundary": POST_MVP_NEXT_BOUNDARY,
        },
        "ordered_work": [{"target": target} for target in ordered_targets],
        "quality_failure_taxonomy_seed": [f"failure_{index}" for index in range(taxonomy_count)],
        "readiness": {
            "post_mvp_quality_iteration_plan_completed": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
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
            "next_boundary": POST_MVP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "next_recommended_issue": "Stage B MIDI-to-solo quality rubric baseline",
    }


class StageBMidiToSoloQualityRubricBaselineTest(unittest.TestCase):
    def test_builds_quality_rubric_without_quality_claim(self) -> None:
        report = build_quality_rubric_baseline_report(
            post_mvp_quality_plan=post_mvp_quality_plan(),
            output_dir=Path("outputs/quality_rubric"),
            issue_number=746,
        )
        summary = validate_quality_rubric_baseline_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            min_rubric_item_count=8,
            require_candidate_labeling_ready=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["quality_rubric_baseline_completed"])
        self.assertTrue(summary["candidate_failure_labeling_ready"])
        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertEqual(summary["rubric_item_count"], 8)
        self.assertGreaterEqual(summary["required_metric_group_count"], 20)
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo candidate failure labeling",
        )

    def test_rejects_wrong_source_boundary(self) -> None:
        source = post_mvp_quality_plan()
        source["boundary"] = "wrong_boundary"
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )

    def test_rejects_wrong_selected_target(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(selected_target="repair_sweep"),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(quality_claim=True),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )

    def test_rejects_incomplete_plan_inputs(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(ordered_work_count=3),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(taxonomy_count=6),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )

    def test_rejects_incomplete_outside_soloing_repair_context(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_ready=False),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_wav_count=5),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_risk_after=1),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=746,
            )

    def test_rejects_missing_required_rubric_item(self) -> None:
        report = build_quality_rubric_baseline_report(
            post_mvp_quality_plan=post_mvp_quality_plan(),
            output_dir=Path("outputs/quality_rubric"),
            issue_number=746,
        )
        report["rubric_items"] = [
            item for item in report["rubric_items"] if item["id"] != "weak_chord_tone_landing"
        ]
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            validate_quality_rubric_baseline_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                min_rubric_item_count=7,
                require_candidate_labeling_ready=True,
                require_no_quality_claim=True,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_quality_rubric_baseline")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_candidate_failure_labeling")
        self.assertEqual(SELECTED_TARGET, "candidate_failure_labeling")


if __name__ == "__main__":
    unittest.main()
