from __future__ import annotations

import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

import pretty_midi

from scripts.audit_stage_b_midi_to_solo_final_status import BRIDGE_SOURCE_CONTEXT_KEYS
from scripts.build_stage_b_midi_to_solo_targeted_quality_repair_listening_review_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloTargetedQualityRepairListeningReviewPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
)
from scripts.render_stage_b_midi_to_solo_targeted_quality_repair_audio import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate([60, 62, 64, 67] * 6):
        start = index * 0.5
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.25)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


def write_wav(path: Path, *, sample_rate: int = 44100) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * (sample_rate // 10))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_rate": sample_rate,
        "frame_count": sample_rate // 10,
        "duration_seconds": 0.1,
    }


def audio_package_report(root: Path, *, quality_claim: bool = False) -> dict:
    rendered = []
    for index in range(1, 7):
        midi_path = root / f"candidate_{index}.mid"
        wav_path = root / f"candidate_{index}.wav"
        write_midi(midi_path)
        wav_file = write_wav(wav_path)
        rendered.append(
            {
                "candidate_index": index,
                "source": "unit_source",
                "rank": index,
                "source_midi_path": str(midi_path),
                "repaired_midi_path": str(midi_path),
                "source_failure_labels": ["rhythmic_monotony", "songlike_melody_not_soloing"],
                "repaired_failure_labels": ["songlike_melody_not_soloing"] if index < 6 else [],
                "changed_pitch_count": index,
                "changed_time_count": 20 + index,
                "note_count": 24,
                "repaired_unique_pitch_count": 4,
                "repaired_dead_air_ratio": 0.0,
                "repaired_max_interval": 7,
                "repaired_max_simultaneous_notes": 1,
                "wav_file": wav_file,
            }
        )
    return {
        "audio_render_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 6,
            "technical_wav_validation": True,
            "targeted_quality_repair_audio_package_completed": True,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "summary": {
            "rendered_audio_file_count": 6,
            "technical_wav_validation": True,
            "sample_rate": 44100,
            "duration_min_seconds": 0.1,
            "duration_max_seconds": 0.1,
            "source_total_failure_label_count": 12,
            "repaired_total_failure_label_count": 5,
            "failure_label_delta": 7,
            "improved_candidate_count": 6,
            "technical_regression_count": 0,
            "source_outside_soloing_repair_evidence_ready": True,
            "source_outside_soloing_repair_source_context_preserved": True,
            "source_outside_soloing_repair_wav_count": 6,
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_source_targeted": False,
            "source_outside_soloing_repair_source_residual_risk_preserved": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_repair_pitch_role_risk_delta": 2,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "audio_review_required": True,
            **SOURCE_CONTEXT,
        },
        "rendered_audio_files": rendered,
    }


class StageBMidiToSoloTargetedQualityRepairListeningReviewPackageTest(unittest.TestCase):
    def test_builds_pending_listening_review_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_package_report(
                audio_package_report=audio_package_report(root),
                output_dir=root / "review_package",
                issue_number=754,
                expected_count=6,
            )
            summary = validate_listening_review_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_review_item_count=6,
                require_package_ready=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["listening_review_package_ready"])
            self.assertEqual(summary["review_item_count"], 6)
            self.assertFalse(summary["validated_review_input"])
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["failure_label_delta"], 7)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertTrue(summary["source_outside_soloing_repair_source_context_preserved"])
            self.assertEqual(summary["source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary["source_outside_soloing_repair_source_objective_pitch_role_risk_count"], 5
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_before"], 5
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_after"], 2
            )
            self.assertEqual(summary["source_outside_soloing_repair_source_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_source_targeted"])
            self.assertTrue(summary["source_outside_soloing_repair_source_residual_risk_preserved"])
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            for key in BRIDGE_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertEqual(
                summary["next_recommended_issue"],
                "Stage B MIDI-to-solo targeted quality repair listening review input guard source-context refresh",
            )

    def test_rejects_audio_package_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(
                StageBMidiToSoloTargetedQualityRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=audio_package_report(root, quality_claim=True),
                    output_dir=root / "review_package",
                    issue_number=754,
                    expected_count=6,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = audio_package_report(root)
            source["summary"]["repaired_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(
                StageBMidiToSoloTargetedQualityRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=source,
                    output_dir=root / "review_package",
                    issue_number=754,
                    expected_count=6,
                )

    def test_rejects_missing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = audio_package_report(root)
            source["summary"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(
                StageBMidiToSoloTargetedQualityRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=source,
                    output_dir=root / "review_package",
                    issue_number=1008,
                    expected_count=6,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_listening_review_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_listening_review_input_guard")


if __name__ == "__main__":
    unittest.main()
