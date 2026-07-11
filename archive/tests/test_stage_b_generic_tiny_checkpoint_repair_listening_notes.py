from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_generic_tiny_checkpoint_repair_listening_notes import (
    StageBGenericTinyCheckpointRepairListeningNotesError,
    build_listening_note,
    build_notes_report,
    validate_notes_report,
)


def candidate(rank: int = 1) -> dict:
    return {
        "review_rank": rank,
        "sample_seed": 42 + rank,
        "sample_index": rank,
        "package_midi_path": f"outputs/review/rank_{rank:02d}.mid",
        "dead_air_ratio": 0.5,
        "phrase_coverage_ratio": 0.8,
        "chord_tone_ratio": 0.4,
        "unique_pitch_count": 6,
        "max_simultaneous_notes": 2,
        "adjacent_repeated_pitch_ratio": 0.0,
        "direction_change_ratio": 0.8,
        "root_tone_ratio": 0.1,
        "tension_ratio": 0.2,
    }


def notes_report(*, ready: bool = True, filled: bool = False, musical_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_notes",
            "listening_notes_ready": ready,
            "human_review_filled": filled,
            "musical_quality_claimed": musical_claim,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_listening_fill",
            "critical_user_input_required": False,
        },
        "source_summary": {
            "candidate_count": 5,
            "failed_candidate_count": 1,
        },
        "listening_notes": {
            "candidate_count": 5,
            "status": "pending_human_review",
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint repair listening fill",
    }


class StageBGenericTinyCheckpointRepairListeningNotesTest(unittest.TestCase):
    def test_builds_pending_note_fields(self) -> None:
        note = build_listening_note(candidate())

        self.assertEqual(note["human_review"]["status"], "pending")
        self.assertEqual(note["human_review"]["keep_decision"], "")
        self.assertEqual(note["objective_context"]["dead_air_ratio"], 0.5)
        self.assertIn("rank_01.mid", note["midi_path"])

    def test_accepts_ready_pending_notes_without_quality_claim(self) -> None:
        summary = validate_notes_report(
            notes_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_listening_notes",
            require_listening_notes_ready=True,
            require_pending_human_review=True,
            require_no_musical_quality_claim=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertEqual(summary["notes_candidate_count"], 5)
        self.assertEqual(summary["notes_status"], "pending_human_review")
        self.assertFalse(summary["human_review_filled"])
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_filled_human_review(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairListeningNotesError):
            validate_notes_report(
                notes_report(filled=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_listening_notes",
                require_listening_notes_ready=True,
                require_pending_human_review=True,
                require_no_musical_quality_claim=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_musical_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairListeningNotesError):
            validate_notes_report(
                notes_report(musical_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_listening_notes",
                require_listening_notes_ready=True,
                require_pending_human_review=True,
                require_no_musical_quality_claim=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_ready_notes_to_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(issue_number=401, min_candidate_count=5)
            report = build_notes_report(
                run_dir=root,
                review_package_report_path=root / "review_package.json",
                review_package_report={
                    "review_package": {
                        "candidate_count": 5,
                        "failed_candidate_count": 1,
                        "midi_dir": "outputs/review/midi",
                    }
                },
                notes=[build_listening_note(candidate(index)) for index in range(1, 6)],
                args=args,
            )

        self.assertTrue(report["readiness"]["listening_notes_ready"])
        self.assertFalse(report["readiness"]["musical_quality_claimed"])
        self.assertEqual(report["decision"]["next_boundary"], "stage_b_generic_tiny_checkpoint_repair_listening_fill")


if __name__ == "__main__":
    unittest.main()
