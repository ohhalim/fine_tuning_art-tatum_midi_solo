from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package import (
    BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError,
    build_audio_render_package,
    validate_audio_render_package,
)


def fake_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00`MTrk\x00\x00\x00\x04\x00\xff/\x00")


def fake_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def candidate(root: Path, *, target_qualified: bool = True) -> dict:
    midi_path = root / "stage_b_sample_7.mid"
    fake_midi(midi_path)
    return {
        "sample_index": 7,
        "sample_seed": 86,
        "interval_cap": 5,
        "midi_path": str(midi_path),
        "target_qualified": target_qualified,
        "note_count": 12,
        "phrase_coverage_ratio": 1.0,
        "tail_empty_steps": 1,
        "postprocess_removal_ratio": 0.35,
        "soft_failure_reasons": ["tail_empty_above_decision_target"],
        "sparse_phrase_metrics": {
            "gap_ratio_to_window": 0.2188,
            "max_internal_gap_beats": 0.5,
            "adjacent_repeat_count": 0,
            "evidence_flags": [],
        },
        "midi_note_audit": {
            "pitch_min": 58,
            "pitch_max": 66,
            "pitch_span": 8,
            "max_abs_interval": 8,
            "large_interval_ratio": 0.0,
            "severe_interval_count": 0,
            "pitch_sequence": [58, 60, 62, 66],
            "intervals": [2, 2, 4],
        },
    }


def sweep_report(root: Path, *, target_passed: bool = True, quality_claimed: bool = False) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep_v1"
        ),
        "run_dir": str(root / "sweep"),
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "sparse_phrase_repair_target_passed": target_passed,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": quality_claimed,
            "quality_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
        },
        "sparse_phrase_repair": {
            "target_passed": target_passed,
            "target_qualified_count": 1 if target_passed else 0,
            "candidate_count": 1,
            "ranked_candidates": [candidate(root, target_qualified=target_passed)],
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageTest(
    unittest.TestCase
):
    def test_builds_ready_package_for_sparse_phrase_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            fake_executable(renderer)
            soundfont.write_bytes(b"sf2")
            report = build_audio_render_package(
                sweep_report(root),
                output_dir=root / "render_package",
                requested_renderer="fluidsynth",
                soundfont_path=str(soundfont),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
                max_review_items=3,
            )
            summary = validate_audio_render_package(
                report,
                expected_boundary=BOUNDARY,
                expected_status="ready_for_local_render",
                min_planned_outputs=1,
                require_target_qualified=True,
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["render_status"], "ready_for_local_render")
            self.assertTrue(summary["soundfont_exists"])
            self.assertEqual(summary["planned_audio_output_count"], 1)
            self.assertEqual(report["review_items"][0]["interval_cap"], 5)
            self.assertEqual(report["review_items"][0]["sparse_phrase_metrics"]["gap_ratio_to_window"], 0.2188)
            self.assertTrue(report["planned_audio_outputs"][0]["render_command"])
            self.assertIn(str(soundfont), report["planned_audio_outputs"][0]["render_command"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_unqualified_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(
                StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError
            ):
                build_audio_render_package(
                    sweep_report(Path(temp_dir), target_passed=False),
                    output_dir=Path(temp_dir) / "render_package",
                )

    def test_rejects_quality_claimed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(
                StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseAudioRenderPackageError
            ):
                build_audio_render_package(
                    sweep_report(Path(temp_dir), quality_claimed=True),
                    output_dir=Path(temp_dir) / "render_package",
                )


if __name__ == "__main__":
    unittest.main()
