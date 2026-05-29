from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package import (
    DurationCoverageFillAudioReviewPackageError,
    build_audio_review_package,
    validate_audio_review_package,
)


def write_midi(path: Path, pitch: int) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    piano.notes.append(pretty_midi.Note(velocity=76, pitch=pitch, start=0.0, end=0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def human_audio_boundary(root: Path) -> dict:
    source_path = root / "source.mid"
    selected_path = root / "selected.mid"
    context_path = root / "selected_context.mid"
    write_midi(source_path, 60)
    write_midi(selected_path, 62)
    write_midi(context_path, 64)
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary_v1",
        "review_items": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
                "prior_decision": "needs_duration_coverage_fill",
                "midi_path": str(source_path),
                "context_midi_path": "",
                "metric_summary": {
                    "note_count": 1,
                    "focused_note_count": 1,
                    "dead_air_ratio": 0.5,
                },
                "note_signature_count": 1,
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
                "prior_decision": "keep",
                "midi_path": str(selected_path),
                "context_midi_path": str(context_path),
                "metric_summary": {
                    "note_count": 1,
                    "focused_note_count": 1,
                    "dead_air_ratio": 0.25,
                },
                "note_signature_count": 1,
            },
        ],
    }


def review_fill_guard() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_fill_v1",
        "candidate_id": "duration_fill_candidate",
        "fill_status": "pending_review_input",
        "claim_boundary": {
            "preference_claimed": False,
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillAudioReviewPackageTest(unittest.TestCase):
    def test_builds_package_with_required_files_and_template(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_audio_review_package(
                human_audio_boundary(root),
                review_fill_guard(),
                output_dir=root / "package",
            )
            summary = validate_audio_review_package(
                report,
                expected_candidate_id="duration_fill_candidate",
                require_files_exist=True,
                require_no_preference=True,
            )

            self.assertEqual(summary["review_item_count"], 2)
            self.assertEqual(summary["package_status"], "ready_for_external_review_input")
            self.assertEqual(summary["audio_render_status"], "not_rendered_by_harness")
            self.assertFalse(summary["preference_claimed"])
            self.assertEqual(summary["required_file_count"], 3)
            self.assertEqual(report["review_input_template"]["candidate_id"], "duration_fill_candidate")
            self.assertGreater(len(report["review_items"][0]["midi_file"]["sha256"]), 10)

    def test_rejects_missing_selected_context_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            boundary = human_audio_boundary(root)
            selected = boundary["review_items"][1]
            Path(selected["context_midi_path"]).unlink()

            with self.assertRaises(DurationCoverageFillAudioReviewPackageError):
                build_audio_review_package(
                    boundary,
                    review_fill_guard(),
                    output_dir=root / "package",
                )

    def test_rejects_preference_already_claimed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = review_fill_guard()
            guard["claim_boundary"]["preference_claimed"] = True

            with self.assertRaises(DurationCoverageFillAudioReviewPackageError):
                build_audio_review_package(
                    human_audio_boundary(root),
                    guard,
                    output_dir=root / "package",
                )


if __name__ == "__main__":
    unittest.main()
