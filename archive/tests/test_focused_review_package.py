from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_focused_review_package import (
    FocusedReviewPackageError,
    build_focused_review_package,
    selected_review_candidates,
)


def sample_candidate(candidate_id: str, decision: str, midi_path: Path, context_path: Path) -> dict:
    return {
        "candidate_id": candidate_id,
        "review_metadata": {
            "mode": "data_motif_rhythm_phrase_variation",
            "review_rank": 1,
            "sample_index": 3,
        },
        "review_files": {
            "midi_path": str(midi_path),
            "context_midi_path": str(context_path),
            "source_midi_path": "outputs/source.mid",
        },
        "source_metrics": {
            "note_count": 63,
            "unique_pitch_count": 28,
            "tension_ratio": 0.413,
        },
        "listening": {
            "status": "reviewed",
            "phrase_quality": "phrase",
            "timing": "acceptable",
            "chord_fit": "fits",
            "issues": [],
            "decision": decision,
            "notes": "proxy note",
        },
        "objective_review": {
            "objective_bucket": "clean",
            "objective_flags": [],
            "objective_reviewable": True,
        },
    }


class FocusedReviewPackageTest(unittest.TestCase):
    def test_selected_review_candidates_filters_decision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            keep = sample_candidate("keep_candidate", "keep", root / "keep.mid", root / "keep_context.mid")
            followup = sample_candidate(
                "followup_candidate",
                "needs_followup",
                root / "followup.mid",
                root / "followup_context.mid",
            )

            selected = selected_review_candidates({"candidates": [followup, keep]}, decision="keep")

        self.assertEqual([candidate["candidate_id"] for candidate in selected], ["keep_candidate"])

    def test_selected_review_candidates_requires_listening_object(self) -> None:
        with self.assertRaises(FocusedReviewPackageError):
            selected_review_candidates({"candidates": [{"candidate_id": "bad"}]}, decision="keep")

    def test_build_focused_review_package_copies_keep_files_and_preserves_objective_notes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            midi_path = root / "candidate.mid"
            context_path = root / "candidate_context.mid"
            midi_path.write_bytes(b"midi")
            context_path.write_bytes(b"context")
            review_notes = {
                "review_notes_path": "outputs/review_notes.json",
                "source_review_manifest": "outputs/review_manifest.json",
                "source_objective_midi_review_report": "outputs/objective.json",
                "candidates": [
                    sample_candidate("keep_candidate", "keep", midi_path, context_path),
                    sample_candidate("followup_candidate", "needs_followup", midi_path, context_path),
                ],
            }
            objective_report = {
                "candidates": [
                    {
                        "candidate_id": "keep_candidate",
                        "first_16_notes": [{"pitch": 70, "start_beats": 0.0}],
                    }
                ]
            }

            package = build_focused_review_package(
                review_notes,
                output_dir=root / "focused",
                copy_files=True,
                objective_report=objective_report,
            )

            self.assertEqual(package["schema_version"], "stage_b_focused_review_package_v1")
            self.assertEqual(package["source_review_notes"], "outputs/review_notes.json")
            self.assertEqual(package["candidate_count"], 1)
            candidate = package["candidates"][0]
            self.assertEqual(candidate["candidate_id"], "keep_candidate")
            self.assertEqual(candidate["objective_first_16_notes"], [{"pitch": 70, "start_beats": 0.0}])
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())


if __name__ == "__main__":
    unittest.main()
