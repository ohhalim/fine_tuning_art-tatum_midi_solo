from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import BOUNDARY as DELIVERY_BOUNDARY
from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    BOUNDARY as RUBRIC_BOUNDARY,
    NEXT_BOUNDARY as RUBRIC_NEXT_BOUNDARY,
    SELECTED_TARGET as RUBRIC_SELECTED_TARGET,
    build_quality_rubric_baseline_report,
)
from scripts.label_stage_b_midi_to_solo_candidate_failures import (
    BOUNDARY,
    REPAIR_NEXT_BOUNDARY,
    REPAIR_TARGET,
    StageBMidiToSoloCandidateFailureLabelingError,
    build_candidate_failure_labeling_report,
    validate_candidate_failure_labeling_report,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.5
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.25)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


def post_mvp_quality_plan() -> dict:
    return {
        "boundary": POST_MVP_BOUNDARY,
        "selected_next_target": {
            "selected_target": POST_MVP_SELECTED_TARGET,
            "selected_next_boundary": POST_MVP_NEXT_BOUNDARY,
        },
        "ordered_work": [
            {"target": "quality_rubric_baseline"},
            {"target": "candidate_failure_labeling"},
            {"target": "targeted_quality_repair_sweep"},
            {"target": "audio_review_package"},
        ],
        "quality_failure_taxonomy_seed": [f"failure_{index}" for index in range(7)],
        "post_mvp_status": {
            "technical_mvp_complete": True,
            "local_review_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "post_mvp_quality_iteration_plan_completed": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
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
    }


def rubric_baseline(root: Path, *, quality_claim: bool = False) -> dict:
    report = build_quality_rubric_baseline_report(
        post_mvp_quality_plan=post_mvp_quality_plan(),
        output_dir=root / "rubric",
        issue_number=746,
    )
    if quality_claim:
        report["readiness"]["midi_to_solo_musical_quality_claimed"] = True
    return report


def delivery_package(paths: list[Path], *, quality_claim: bool = False) -> dict:
    return {
        "boundary": DELIVERY_BOUNDARY,
        "delivery_package": {
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
        },
        "artifact_manifest": {
            "cli_repaired_midi_candidates": [
                {
                    "rank": 1,
                    "repaired_midi_path": str(paths[0]),
                    "note_count": 16,
                    "unique_pitch_count": 5,
                    "dead_air_ratio": 0.0,
                    "objective_supported": True,
                },
                {
                    "rank": 2,
                    "repaired_midi_path": str(paths[1]),
                    "note_count": 16,
                    "unique_pitch_count": 5,
                    "dead_air_ratio": 0.0,
                    "objective_supported": True,
                },
            ],
            "changed_ratio_repair_audio_candidates": [
                {
                    "rank": 1,
                    "repaired_midi_path": str(paths[2]),
                    "repaired_unique_pitch_count": 5,
                    "pitch_changed_ratio": 0.3,
                    "repaired_max_interval": 4,
                }
            ],
        },
        "readiness": {
            "mvp_delivery_package_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
    }


class StageBMidiToSoloCandidateFailureLabelingTest(unittest.TestCase):
    def test_labels_candidate_failures_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"candidate_{index}.mid" for index in range(3)]
            for path in paths:
                write_midi(path, [60, 62, 64, 65] * 4)
            report = build_candidate_failure_labeling_report(
                rubric_baseline=rubric_baseline(root),
                mvp_delivery_package=delivery_package(paths),
                output_dir=root / "labels",
                issue_number=748,
            )
            summary = validate_candidate_failure_labeling_report(
                report,
                expected_boundary=BOUNDARY,
                min_candidate_count=3,
                require_labeling_completed=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["candidate_failure_labeling_completed"])
            self.assertEqual(summary["candidate_count"], 3)
            self.assertGreater(summary["failed_candidate_count"], 0)
            self.assertGreater(summary["failure_label_type_count"], 0)
            self.assertGreaterEqual(summary["not_evaluable_label_type_count"], 2)
            self.assertEqual(summary["selected_target"], REPAIR_TARGET)
            self.assertEqual(summary["next_boundary"], REPAIR_NEXT_BOUNDARY)
            self.assertTrue(summary["targeted_quality_repair_sweep_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_delivery_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"candidate_{index}.mid" for index in range(3)]
            for path in paths:
                write_midi(path, [60, 62, 64, 65] * 4)
            with self.assertRaises(StageBMidiToSoloCandidateFailureLabelingError):
                build_candidate_failure_labeling_report(
                    rubric_baseline=rubric_baseline(root),
                    mvp_delivery_package=delivery_package(paths, quality_claim=True),
                    output_dir=root / "labels",
                    issue_number=748,
                )

    def test_rejects_rubric_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"candidate_{index}.mid" for index in range(3)]
            for path in paths:
                write_midi(path, [60, 62, 64, 65] * 4)
            with self.assertRaises(StageBMidiToSoloCandidateFailureLabelingError):
                build_candidate_failure_labeling_report(
                    rubric_baseline=rubric_baseline(root, quality_claim=True),
                    mvp_delivery_package=delivery_package(paths),
                    output_dir=root / "labels",
                    issue_number=748,
                )

    def test_rejects_missing_candidate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"candidate_{index}.mid" for index in range(3)]
            for path in paths[:2]:
                write_midi(path, [60, 62, 64, 65] * 4)
            with self.assertRaises(StageBMidiToSoloCandidateFailureLabelingError):
                build_candidate_failure_labeling_report(
                    rubric_baseline=rubric_baseline(root),
                    mvp_delivery_package=delivery_package(paths),
                    output_dir=root / "labels",
                    issue_number=748,
                )

    def test_rejects_wrong_rubric_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = rubric_baseline(root)
            source["boundary"] = "wrong_boundary"
            with self.assertRaises(StageBMidiToSoloCandidateFailureLabelingError):
                build_candidate_failure_labeling_report(
                    rubric_baseline=source,
                    mvp_delivery_package=delivery_package([root / "a.mid"] * 3),
                    output_dir=root / "labels",
                    issue_number=748,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_candidate_failure_labeling")
        self.assertEqual(RUBRIC_BOUNDARY, "stage_b_midi_to_solo_quality_rubric_baseline")
        self.assertEqual(RUBRIC_NEXT_BOUNDARY, "stage_b_midi_to_solo_candidate_failure_labeling")
        self.assertEqual(RUBRIC_SELECTED_TARGET, "candidate_failure_labeling")


if __name__ == "__main__":
    unittest.main()
