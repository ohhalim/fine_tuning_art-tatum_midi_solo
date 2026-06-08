from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.export_stage_b_midi_to_solo_model_conditioned_input_path_candidates import (
    BOUNDARY,
    MODEL_CONDITIONED_SOURCE,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathCandidateExportError,
    build_candidate_export_report,
    validate_candidate_export_report,
)
from scripts.probe_stage_b_midi_to_solo_model_conditioned_input_path import (
    BOUNDARY as PROBE_BOUNDARY,
    NEXT_BOUNDARY as PROBE_NEXT_BOUNDARY,
)


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"midi")
    return str(path)


def probe_report() -> dict:
    return {
        "boundary": PROBE_BOUNDARY,
        "replacement_decision": {
            "selected_next_boundary": PROBE_NEXT_BOUNDARY,
        },
        "alignment_source": {
            "phrase_bank_cli_technical_path_completed": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "readiness": {
            "model_conditioned_input_path_probe_completed": True,
            "model_conditioned_candidate_source_available": True,
            "model_conditioned_audio_technical_path_available": True,
            "same_input_context_as_fallback": True,
            "model_conditioned_ranked_input_path_contract_matched": False,
            "fallback_replacement_ready": False,
            "candidate_export_required": True,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": PROBE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def model_direct_repair_report(root: Path) -> dict:
    midi_paths = [touch_file(root, f"samples/sample_{index}.mid") for index in range(1, 4)]
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
        "generation_report_path": "generation_probe/report.json",
        "repair_config": {
            "generation_source": MODEL_CONDITIONED_SOURCE,
        },
        "repaired_generation_summary": {
            "sample_count": 3,
            "strict_valid_sample_count": 3,
            "grammar_gate_sample_count": 3,
            "min_postprocess_note_count": 24,
            "max_postprocess_note_count": 24,
            "midi_paths": midi_paths,
        },
        "context_summary": {
            "context_bars": 8,
            "bpm": 120,
            "chord_progression": ["Cmaj7", "F7", "G7", "Cmaj7"],
        },
        "readiness": {
            "direct_generation_review_gate_passed": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def sample_row(root: Path, index: int, *, note_count: int = 24) -> dict:
    midi_path = touch_file(root, f"generation/sample_{index}.mid")
    return {
        "sample_index": index,
        "sample_seed": 490 + index,
        "midi_path": midi_path,
        "valid": True,
        "strict_valid": True,
        "grammar_gate_passed": True,
        "metrics": {
            "note_count": note_count,
            "unique_pitch_count": 12 + index,
            "unique_pitch_class_count": 7,
            "max_simultaneous_notes": 1,
            "dead_air_ratio": 0.4 + (index * 0.01),
            "phrase_coverage_ratio": 0.9,
            "chord_tone_ratio": 0.7,
        },
        "collapse": {
            "repeated_pitch_ratio": 0.1,
            "postprocess_removal_ratio": 0.0,
        },
        "phrase_contour": {
            "direction_change_ratio": 0.5,
            "stepwise_motion_ratio": 0.2,
            "leap_motion_ratio": 0.8,
        },
        "pitch_roles": {
            "non_chord_tone_ratio": 0.3,
        },
        "temporal_coverage": {
            "onset_coverage_ratio": 0.18,
            "sustained_coverage_ratio": 0.6,
            "position_span_ratio": 0.9,
        },
    }


def generation_report(root: Path, *, note_count: int = 24) -> dict:
    return {
        "generation_mode": "constrained",
        "passed_grammar_gate": True,
        "passed_strict_review_gate": True,
        "samples": [sample_row(root, index, note_count=note_count) for index in range(1, 4)],
    }


class StageBMidiToSoloModelConditionedInputPathCandidateExportTest(unittest.TestCase):
    def test_exports_model_conditioned_candidates_through_ranked_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_candidate_export_report(
                probe_report=probe_report(),
                model_direct_repair_report=model_direct_repair_report(root),
                model_direct_generation_report=generation_report(root),
                output_dir=root / "out",
                issue_number=624,
                export_top_midi_count=3,
            )
            summary = validate_candidate_export_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_exported_candidates=3,
                require_ranked_export_contract=True,
                require_audio_render_required=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["model_conditioned_input_path_candidate_export_completed"])
        self.assertTrue(summary["ranked_midi_candidates_exported"])
        self.assertTrue(summary["model_conditioned_ranked_input_path_contract_matched"])
        self.assertTrue(summary["fallback_replacement_candidate_export_ready"])
        self.assertFalse(summary["fallback_replacement_ready"])
        self.assertTrue(summary["candidate_audio_render_required"])
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertEqual(summary["exported_candidate_count"], 3)
        self.assertEqual(summary["best_max_simultaneous_notes"], 1)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_probe_when_candidate_export_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = probe_report()
            probe["readiness"]["candidate_export_required"] = False
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathCandidateExportError):
                build_candidate_export_report(
                    probe_report=probe,
                    model_direct_repair_report=model_direct_repair_report(root),
                    model_direct_generation_report=generation_report(root),
                    output_dir=root / "out",
                    issue_number=624,
                    export_top_midi_count=3,
                )

    def test_rejects_generation_sample_below_note_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathCandidateExportError):
                build_candidate_export_report(
                    probe_report=probe_report(),
                    model_direct_repair_report=model_direct_repair_report(root),
                    model_direct_generation_report=generation_report(root, note_count=23),
                    output_dir=root / "out",
                    issue_number=624,
                    export_top_midi_count=3,
                )

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = probe_report()
            probe["readiness"]["human_audio_preference_claimed"] = True
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathCandidateExportError):
                build_candidate_export_report(
                    probe_report=probe,
                    model_direct_repair_report=model_direct_repair_report(root),
                    model_direct_generation_report=generation_report(root),
                    output_dir=root / "out",
                    issue_number=624,
                    export_top_midi_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package")


if __name__ == "__main__":
    unittest.main()
