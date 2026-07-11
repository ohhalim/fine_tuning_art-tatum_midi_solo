from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_model_conditioned_input_path_replacement import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError,
    build_replacement_consolidation_report,
    validate_replacement_consolidation_report,
)
from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (
    BOUNDARY as CANDIDATE_EXPORT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_audio import (
    BOUNDARY as AUDIO_RENDER_BOUNDARY,
    NEXT_BOUNDARY as AUDIO_RENDER_NEXT_BOUNDARY,
)


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return str(path)


def candidate_export_report(root: Path) -> dict:
    top = [
        {
            "rank": index,
            "export_midi_path": touch_file(root, f"midi/rank_{index}.mid"),
        }
        for index in range(1, 4)
    ]
    return {
        "boundary": CANDIDATE_EXPORT_BOUNDARY,
        "summary": {
            "exported_candidate_count": 3,
            "best_note_count": 24,
            "best_unique_pitch_count": 20,
            "best_max_simultaneous_notes": 1,
            "best_dead_air_ratio": 0.65,
        },
        "probe_source": {
            "phrase_bank_cli_technical_path_completed": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "top_candidates": top,
        "readiness": {
            "model_conditioned_input_path_candidate_export_completed": True,
            "ranked_midi_candidates_exported": True,
            "model_conditioned_ranked_input_path_contract_matched": True,
            "fallback_replacement_candidate_export_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def audio_render_report(root: Path, midi_paths: list[str], *, quality_claim: bool = False) -> dict:
    files = [
        {
            "rank": index,
            "source_midi_path": midi_paths[index - 1],
            "wav_file": {
                "path": touch_file(root, f"audio/rank_{index}.wav"),
                "exists": True,
                "sample_rate": 44100,
                "frame_count": 44100,
                "duration_seconds": 1.0 + index,
            },
        }
        for index in range(1, 4)
    ]
    return {
        "audio_render_boundary": {
            "boundary": AUDIO_RENDER_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "model_conditioned_ranked_audio_render_completed": True,
            "fallback_replacement_candidate_export_ready": True,
            "fallback_replacement_technical_path_ready": True,
            "fallback_replacement_ready": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": quality_claim,
            "musical_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": AUDIO_RENDER_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "candidate_export_source": {
            "phrase_bank_cli_technical_path_completed": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "rendered_audio_files": files,
    }


class StageBMidiToSoloModelConditionedInputPathReplacementConsolidationTest(unittest.TestCase):
    def test_consolidates_ranked_midi_and_wav_technical_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = candidate_export_report(root)
            midi_paths = [item["export_midi_path"] for item in candidate["top_candidates"]]
            report = build_replacement_consolidation_report(
                candidate_export_report=candidate,
                audio_render_report=audio_render_report(root, midi_paths),
                output_dir=root / "out",
                issue_number=628,
                expected_count=3,
            )
            summary = validate_replacement_consolidation_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_technical_replacement=True,
                require_listening_review_package=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["model_conditioned_input_path_replacement_consolidated"])
        self.assertTrue(summary["model_conditioned_input_to_ranked_midi_completed"])
        self.assertTrue(summary["model_conditioned_input_to_ranked_wav_completed"])
        self.assertTrue(summary["fallback_replacement_technical_path_ready"])
        self.assertTrue(summary["fallback_replacement_ready"])
        self.assertTrue(summary["listening_review_package_required"])
        self.assertEqual(summary["exported_candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_mismatched_audio_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = candidate_export_report(root)
            wrong_paths = [touch_file(root, f"wrong/{index}.mid") for index in range(1, 4)]
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError):
                build_replacement_consolidation_report(
                    candidate_export_report=candidate,
                    audio_render_report=audio_render_report(root, wrong_paths),
                    output_dir=root / "out",
                    issue_number=628,
                    expected_count=3,
                )

    def test_rejects_audio_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = candidate_export_report(root)
            midi_paths = [item["export_midi_path"] for item in candidate["top_candidates"]]
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathReplacementConsolidationError):
                build_replacement_consolidation_report(
                    candidate_export_report=candidate,
                    audio_render_report=audio_render_report(root, midi_paths, quality_claim=True),
                    output_dir=root / "out",
                    issue_number=628,
                    expected_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package")


if __name__ == "__main__":
    unittest.main()
