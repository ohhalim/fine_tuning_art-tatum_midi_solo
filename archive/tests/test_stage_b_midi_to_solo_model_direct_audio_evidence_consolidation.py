from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_model_direct_audio_evidence import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectAudioEvidenceConsolidationError,
    build_model_direct_audio_evidence_report,
    validate_model_direct_audio_evidence_report,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def objective_report(root: Path, *, strict_count: int = 3, quality_claim: bool = False) -> dict:
    midi_paths = [touch(root / f"stage_b_sample_{index}.mid") for index in range(1, 4)]
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
        "readiness": {
            "boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
            "direct_generation_review_gate_passed": True,
            "model_direct_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
        },
        "repair_result": {
            "previous_valid_sample_count": 0,
            "previous_strict_valid_sample_count": 0,
        },
        "repaired_generation_summary": {
            "sample_count": 3,
            "valid_sample_count": 3,
            "strict_valid_sample_count": strict_count,
            "avg_postprocess_removal_ratio": 0.0,
            "collapse_warning_sample_rate": 0.0,
            "midi_paths": midi_paths,
        },
    }


def audio_report(root: Path, *, rendered_count: int = 3) -> dict:
    wav_paths = [touch(root / f"model_direct_sample_{index:02d}.wav") for index in range(1, rendered_count + 1)]
    return {
        "source_boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
        "audio_render_boundary": {
            "boundary": "stage_b_midi_to_solo_model_direct_audio_render_package",
            "render_attempted": True,
            "rendered_audio_file_count": rendered_count,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "model_direct_generation_quality_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation",
            "critical_user_input_required": False,
        },
        "rendered_audio_files": [
            {
                "sample_index": index,
                "wav_file": {
                    "path": path,
                    "sample_rate": 44100,
                    "duration_seconds": 20.0 + index,
                },
            }
            for index, path in enumerate(wav_paths, start=1)
        ],
    }


class StageBMidiToSoloModelDirectAudioEvidenceConsolidationTest(unittest.TestCase):
    def test_consolidates_objective_and_audio_evidence_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_model_direct_audio_evidence_report(
                objective_report=objective_report(root / "midi"),
                audio_render_report=audio_report(root / "audio"),
                output_dir=root / "out",
                issue_number=503,
            )
            summary = validate_model_direct_audio_evidence_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_technical_path=True,
                require_no_quality_claim=True,
                min_midi_count=3,
                min_wav_count=3,
            )

            self.assertTrue(summary["model_direct_objective_gate_passed"])
            self.assertTrue(summary["model_direct_audio_render_completed"])
            self.assertTrue(summary["model_direct_midi_to_wav_technical_path_completed"])
            self.assertEqual(summary["strict_valid_sample_count"], 3)
            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertEqual(summary["sample_rates"], [44100])
            self.assertFalse(summary["model_direct_generation_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_missing_wav_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = audio_report(root / "audio")
            Path(audio["rendered_audio_files"][0]["wav_file"]["path"]).unlink()
            report = build_model_direct_audio_evidence_report(
                objective_report=objective_report(root / "midi"),
                audio_render_report=audio,
                output_dir=root / "out",
                issue_number=503,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectAudioEvidenceConsolidationError):
                validate_model_direct_audio_evidence_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_technical_path=True,
                    require_no_quality_claim=True,
                    min_midi_count=3,
                    min_wav_count=3,
                )

    def test_rejects_incomplete_strict_midi_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_model_direct_audio_evidence_report(
                objective_report=objective_report(root / "midi", strict_count=2),
                audio_render_report=audio_report(root / "audio"),
                output_dir=root / "out",
                issue_number=503,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectAudioEvidenceConsolidationError):
                validate_model_direct_audio_evidence_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_technical_path=True,
                    require_no_quality_claim=True,
                    min_midi_count=3,
                    min_wav_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_direct_audio_evidence_consolidation")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics")


if __name__ == "__main__":
    unittest.main()
