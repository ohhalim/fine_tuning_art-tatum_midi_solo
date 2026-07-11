from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment import (
    BOUNDARY as ALIGNMENT_BOUNDARY,
    NEXT_BOUNDARY as ALIGNMENT_NEXT_BOUNDARY,
    SELECTED_PROBE_TARGET,
)
from scripts.probe_stage_b_midi_to_solo_model_conditioned_input_path import (
    BOUNDARY,
    FALLBACK_SOURCE,
    MODEL_CONDITIONED_SOURCE,
    NEXT_BOUNDARY,
    StageBMidiToSoloModelConditionedInputPathProbeError,
    build_probe_report,
    validate_probe_report,
)


def touch_file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    return str(path)


def alignment_report() -> dict:
    return {
        "boundary": ALIGNMENT_BOUNDARY,
        "alignment_decision": {
            "fallback_replacement_probe_required": True,
        },
        "alignment_source": {
            "phrase_bank_cli_technical_path_completed": True,
            "cli_candidate_count": 3,
            "cli_rendered_audio_file_count": 3,
            "cli_input_context_bars": 228,
            "cli_preference_fill_allowed": False,
        },
        "readiness": {
            "boundary": ALIGNMENT_BOUNDARY,
            "model_conditioned_input_path_quality_alignment_decision_completed": True,
            "model_conditioned_input_path_aligned": False,
            "fallback_replacement_probe_required": True,
            "selected_probe_target": SELECTED_PROBE_TARGET,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": ALIGNMENT_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def fallback_generation_report(root: Path, *, source: str = FALLBACK_SOURCE) -> dict:
    top_candidates = [
        {
            "rank": index,
            "seed": 480 + index,
            "generation_source": source,
            "note_count": 60,
            "unique_pitch_count": 12,
            "max_simultaneous_notes": 1,
            "export_midi_path": touch_file(root, f"fallback/rank_{index}.mid"),
        }
        for index in range(1, 4)
    ]
    return {
        "boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
        "generation_config": {
            "generation_source": source,
            "model_checkpoint_generation_used": False,
        },
        "summary": {
            "candidate_count": 8,
            "qualified_candidate_count": 8,
            "exported_candidate_count": 3,
            "exported_qualified_candidate_count": 3,
            "best_note_count": 60,
            "best_unique_pitch_count": 12,
            "best_max_simultaneous_notes": 1,
        },
        "readiness": {
            "conditioned_generation_probe_completed": True,
            "ranked_midi_candidates_exported": True,
            "midi_to_solo_mvp_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
        },
        "input_context": {
            "bars": 8,
            "bpm": 120,
            "chord_progression": ["Cmaj7", "F7", "G7", "Cmaj7"],
        },
        "top_candidates": top_candidates,
    }


def audio_report(root: Path, *, boundary: str, source_boundary: str, prefix: str) -> dict:
    rendered = [
        {
            "rank": index,
            "wav_file": {
                "path": touch_file(root, f"{prefix}/rank_{index}.wav"),
                "exists": True,
                "size_bytes": 1000,
                "frame_count": 44100,
                "duration_seconds": 1.0,
            },
        }
        for index in range(1, 4)
    ]
    return {
        "source_boundary": source_boundary,
        "rendered_audio_files": rendered,
        "audio_render_boundary": {
            "boundary": boundary,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "midi_to_solo_mvp_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def model_direct_repair_report(root: Path, *, strict_count: int = 3) -> dict:
    midi_paths = [touch_file(root, f"model_direct/sample_{index}.mid") for index in range(1, 4)]
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
        "repair_config": {
            "generation_source": MODEL_CONDITIONED_SOURCE,
            "target_bars": 8,
            "note_groups_per_bar": 3,
            "cap_duration_to_next_position": True,
        },
        "repaired_generation_summary": {
            "sample_count": 3,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 3,
            "min_postprocess_note_count": 24,
            "max_postprocess_note_count": 24,
            "avg_postprocess_removal_ratio": 0.0,
            "avg_onset_coverage_ratio": 0.1875,
            "avg_sustained_coverage_ratio": 0.6354,
            "all_midi_paths_exist": True,
            "midi_paths": midi_paths,
        },
        "context_summary": {
            "context_bars": 8,
            "bpm": 120,
            "chord_progression": ["Cmaj7", "F7", "G7", "Cmaj7"],
        },
        "readiness": {
            "monophonic_overlap_repair_completed": True,
            "direct_generated_midi_written": True,
            "direct_generation_review_gate_passed": True,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


class StageBMidiToSoloModelConditionedInputPathProbeTest(unittest.TestCase):
    def test_probe_records_model_conditioned_evidence_and_export_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_probe_report(
                alignment_report=alignment_report(),
                fallback_generation_report=fallback_generation_report(root),
                fallback_audio_report=audio_report(
                    root,
                    boundary="stage_b_midi_to_solo_candidate_audio_render_package",
                    source_boundary="stage_b_midi_to_solo_conditioned_generation_probe",
                    prefix="fallback_audio",
                ),
                model_direct_repair_report=model_direct_repair_report(root),
                model_direct_audio_report=audio_report(
                    root,
                    boundary="stage_b_midi_to_solo_model_direct_audio_render_package",
                    source_boundary="stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
                    prefix="model_direct_audio",
                ),
                output_dir=Path("outputs/probe"),
                issue_number=622,
                min_count=3,
            )
            summary = validate_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_model_conditioned_evidence=True,
                require_candidate_export=True,
                require_replacement_not_ready=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["model_conditioned_candidate_source_available"])
        self.assertTrue(summary["model_conditioned_audio_technical_path_available"])
        self.assertTrue(summary["same_input_context_as_fallback"])
        self.assertFalse(summary["model_conditioned_ranked_input_path_contract_matched"])
        self.assertFalse(summary["fallback_replacement_ready"])
        self.assertTrue(summary["candidate_export_required"])
        self.assertTrue(summary["missing_ranked_export_contract"])
        self.assertEqual(summary["selected_next_boundary"], NEXT_BOUNDARY)
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_non_fallback_generation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathProbeError):
                build_probe_report(
                    alignment_report=alignment_report(),
                    fallback_generation_report=fallback_generation_report(root, source="checkpoint"),
                    fallback_audio_report=audio_report(
                        root,
                        boundary="stage_b_midi_to_solo_candidate_audio_render_package",
                        source_boundary="stage_b_midi_to_solo_conditioned_generation_probe",
                        prefix="fallback_audio",
                    ),
                    model_direct_repair_report=model_direct_repair_report(root),
                    model_direct_audio_report=audio_report(
                        root,
                        boundary="stage_b_midi_to_solo_model_direct_audio_render_package",
                        source_boundary="stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
                        prefix="model_direct_audio",
                    ),
                    output_dir=Path("outputs/probe"),
                    issue_number=622,
                    min_count=3,
                )

    def test_rejects_model_direct_strict_count_below_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathProbeError):
                build_probe_report(
                    alignment_report=alignment_report(),
                    fallback_generation_report=fallback_generation_report(root),
                    fallback_audio_report=audio_report(
                        root,
                        boundary="stage_b_midi_to_solo_candidate_audio_render_package",
                        source_boundary="stage_b_midi_to_solo_conditioned_generation_probe",
                        prefix="fallback_audio",
                    ),
                    model_direct_repair_report=model_direct_repair_report(root, strict_count=2),
                    model_direct_audio_report=audio_report(
                        root,
                        boundary="stage_b_midi_to_solo_model_direct_audio_render_package",
                        source_boundary="stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
                        prefix="model_direct_audio",
                    ),
                    output_dir=Path("outputs/probe"),
                    issue_number=622,
                    min_count=3,
                )

    def test_rejects_model_direct_audio_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_audio = audio_report(
                root,
                boundary="stage_b_midi_to_solo_model_direct_audio_render_package",
                source_boundary="stage_b_midi_to_solo_model_direct_monophonic_overlap_repair",
                prefix="model_direct_audio",
            )
            model_audio["audio_render_boundary"]["human_audio_preference_claimed"] = True
            with self.assertRaises(StageBMidiToSoloModelConditionedInputPathProbeError):
                build_probe_report(
                    alignment_report=alignment_report(),
                    fallback_generation_report=fallback_generation_report(root),
                    fallback_audio_report=audio_report(
                        root,
                        boundary="stage_b_midi_to_solo_candidate_audio_render_package",
                        source_boundary="stage_b_midi_to_solo_conditioned_generation_probe",
                        prefix="fallback_audio",
                    ),
                    model_direct_repair_report=model_direct_repair_report(root),
                    model_direct_audio_report=model_audio,
                    output_dir=Path("outputs/probe"),
                    issue_number=622,
                    min_count=3,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_probe")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_model_conditioned_input_path_candidate_export")


if __name__ == "__main__":
    unittest.main()
