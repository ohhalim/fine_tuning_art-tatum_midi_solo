from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_mvp_execution import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloMvpExecutionConsolidationError,
    build_execution_consolidation_report,
    validate_execution_consolidation_report,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def reports(root: Path, *, quality_claim: bool = False) -> dict[str, dict]:
    midi_paths = [touch(root / f"rank_{index:02d}.mid") for index in range(1, 4)]
    wav_paths = [touch(root / f"rank_{index:02d}.wav") for index in range(1, 4)]
    return {
        "contract": {
            "boundary": "stage_b_midi_to_solo_mvp_input_contract",
            "target_date": "2026-06-11",
            "output_contract": {
                "candidate_count": 32,
                "export_top_midi_count": 3,
                "target_solo_bars": 8,
            },
            "objective_gate": {
                "min_note_count": 24,
                "min_unique_pitch_count": 8,
                "max_simultaneous_notes": 1,
            },
        },
        "context": {
            "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
            "summary": {"context_event_count": 128},
        },
        "resource": {
            "boundary": "stage_b_midi_to_solo_training_resource_probe",
            "readiness": {"midi_to_solo_training_resource_ready": True},
        },
        "generation": {
            "boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
            "generation_config": {"generation_source": "context_conditioned_fallback"},
            "summary": {
                "candidate_count": 8,
                "exported_candidate_count": 3,
                "exported_qualified_candidate_count": 3,
            },
            "readiness": {"ranked_midi_candidates_exported": True},
            "top_candidates": [{"export_midi_path": path} for path in midi_paths],
        },
        "audio": {
            "audio_render_boundary": {
                "boundary": "stage_b_midi_to_solo_candidate_audio_render_package",
                "rendered_audio_file_count": 3,
                "technical_wav_validation": True,
            },
            "decision": {"next_boundary": "stage_b_midi_to_solo_mvp_execution_consolidation"},
            "rendered_audio_files": [{"wav_file": {"path": path}} for path in wav_paths],
        },
    }


class StageBMidiToSoloMvpExecutionConsolidationTest(unittest.TestCase):
    def test_consolidates_technical_mvp_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            report = build_execution_consolidation_report(
                contract_report=data["contract"],
                context_report=data["context"],
                resource_probe=data["resource"],
                generation_probe=data["generation"],
                audio_render=data["audio"],
                output_dir=Path(tmp) / "out",
                issue_number=491,
            )
            summary = validate_execution_consolidation_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_technical_mvp=True,
                require_no_quality_claim=True,
                min_exported_candidates=3,
                min_rendered_wav_files=3,
            )

            self.assertTrue(summary["technical_execution_path_completed"])
            self.assertTrue(summary["midi_to_solo_technical_mvp_completed"])
            self.assertTrue(summary["input_to_ranked_midi_completed"])
            self.assertTrue(summary["input_to_rendered_audio_completed"])
            self.assertEqual(summary["generation_source"], "context_conditioned_fallback")
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertFalse(summary["model_checkpoint_direct_generation_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])

    def test_rejects_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            Path(data["generation"]["top_candidates"][0]["export_midi_path"]).unlink()
            report = build_execution_consolidation_report(
                contract_report=data["contract"],
                context_report=data["context"],
                resource_probe=data["resource"],
                generation_probe=data["generation"],
                audio_render=data["audio"],
                output_dir=Path(tmp) / "out",
                issue_number=491,
            )
            with self.assertRaises(StageBMidiToSoloMvpExecutionConsolidationError):
                validate_execution_consolidation_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_technical_mvp=True,
                    require_no_quality_claim=True,
                    min_exported_candidates=3,
                    min_rendered_wav_files=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_execution_consolidation")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_generation_repair")


if __name__ == "__main__":
    unittest.main()
