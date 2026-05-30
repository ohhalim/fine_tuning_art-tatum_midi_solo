from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.fill_stage_b_generic_tiny_checkpoint_repair_listening_notes import (
    StageBGenericTinyCheckpointRepairListeningFillError,
    build_listening_fill_report,
    build_review_rows,
    note_items,
    validate_listening_fill_report,
)


def listening_note(rank: int = 1) -> dict:
    return {
        "review_rank": rank,
        "sample_seed": 42 + rank,
        "sample_index": rank,
        "midi_path": f"outputs/review/rank_{rank:02d}.mid",
        "human_review": {
            "status": "pending",
        },
    }


def notes_report(*, filled: bool = False, musical_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_notes",
            "human_review_filled": filled,
            "musical_quality_claimed": musical_claim,
        },
        "listening_notes": {
            "status": "pending_human_review",
            "candidate_count": 2,
            "notes": [listening_note(1), listening_note(2)],
        },
    }


def fill_report(*, review_input_present: bool = False, musical_claim: bool = False) -> dict:
    return {
        "review_input_present": review_input_present,
        "fill_status": "review_input_applied" if review_input_present else "pending_review_input",
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_fill",
            "human_review_filled": review_input_present,
            "pending_without_review_input": not review_input_present,
            "musical_quality_claimed": musical_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_audio_render_package",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
        "listening_fill": {
            "status": "reviewed" if review_input_present else "pending_review_input",
            "candidate_count": 2,
            "keep_count": 0,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint repair audio render package",
    }


class StageBGenericTinyCheckpointRepairListeningFillTest(unittest.TestCase):
    def test_note_items_rejects_already_filled_source(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairListeningFillError):
            note_items(notes_report(filled=True))

    def test_builds_pending_reviews_without_input(self) -> None:
        rows = build_review_rows(note_items(notes_report()), None)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "pending_review_input")
        self.assertEqual(rows[0]["keep_decision"], "pending")

    def test_accepts_pending_fill_without_quality_claim(self) -> None:
        summary = validate_listening_fill_report(
            fill_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_listening_fill",
            require_pending_without_input=True,
            require_no_quality_without_input=True,
            require_objective_auto_progress_allowed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertFalse(summary["review_input_present"])
        self.assertEqual(summary["fill_status"], "pending_review_input")
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_quality_claim_without_input(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairListeningFillError):
            validate_listening_fill_report(
                fill_report(musical_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_listening_fill",
                require_pending_without_input=True,
                require_no_quality_without_input=True,
                require_objective_auto_progress_allowed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_absent_input_to_audio_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(issue_number=403)
            report = build_listening_fill_report(
                run_dir=root,
                listening_notes_report_path=root / "notes.json",
                listening_notes_report=notes_report(),
                review_input=None,
                review_rows=build_review_rows(note_items(notes_report()), None),
                args=args,
            )

        self.assertEqual(report["fill_status"], "pending_review_input")
        self.assertFalse(report["readiness"]["musical_quality_claimed"])
        self.assertEqual(
            report["decision"]["next_boundary"],
            "stage_b_generic_tiny_checkpoint_repair_audio_render_package",
        )


if __name__ == "__main__":
    unittest.main()
