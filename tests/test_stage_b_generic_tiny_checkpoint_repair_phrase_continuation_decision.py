from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation import (
    StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError,
    build_phrase_continuation_decision,
    validate_phrase_continuation_decision,
)


def user_listening_review(*, keep_claim: bool = False, primary_failure: str = "plunk_and_stop") -> dict:
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_user_listening_review_v1",
        "reviewed_audio_files": [{}, {}, {}, {}, {}],
        "user_listening_review": {
            "overall_decision": "reject_all",
            "candidate_decision": "reject",
            "primary_failure": primary_failure,
            "timing": "too_short_or_stiff",
            "phrase": "fragmented",
            "vocabulary": "not_musical",
            "assessment": "all candidates only plunk briefly and end",
            "candidate_reviews": [
                {"decision": "reject", "primary_failure": primary_failure},
                {"decision": "reject", "primary_failure": primary_failure},
                {"decision": "reject", "primary_failure": primary_failure},
                {"decision": "reject", "primary_failure": primary_failure},
                {"decision": "reject", "primary_failure": primary_failure},
            ],
        },
        "claim_boundary": {
            "boundary": "generic_tiny_checkpoint_repair_audio_review_reject_all",
            "human_audio_keep_claimed": keep_claim,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_decision",
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationDecisionTest(unittest.TestCase):
    def test_routes_plunk_and_stop_to_phrase_continuation_sweep(self) -> None:
        report = build_phrase_continuation_decision(
            user_listening_review(),
            output_dir=Path("outputs/phrase_continuation_decision"),
        )
        summary = validate_phrase_continuation_decision(
            report,
            expected_next_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            require_auto_progress_allowed=True,
            require_no_critical_user_input=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["human_audio_keep_claimed"])
        self.assertIn("require_phrase_continuation_after_initial_cell", report["repair_targets"])

    def test_rejects_keep_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError):
            build_phrase_continuation_decision(
                user_listening_review(keep_claim=True),
                output_dir=Path("outputs/phrase_continuation_decision"),
            )

    def test_rejects_non_plunk_failure(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError):
            build_phrase_continuation_decision(
                user_listening_review(primary_failure="fragmented"),
                output_dir=Path("outputs/phrase_continuation_decision"),
            )


if __name__ == "__main__":
    unittest.main()
