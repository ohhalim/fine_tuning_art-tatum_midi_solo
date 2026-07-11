from __future__ import annotations

import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

import pretty_midi

from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION as REVIEW_SCHEMA_VERSION,
    SOURCE_PACKAGE_SCHEMA_CONTEXT_KEYS,
    StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError,
    build_listening_review_package_report,
    validate_listening_review_package_report,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    SOURCE_SCHEMA_CONTEXT_KEYS as SOURCE_AUDIO_SCHEMA_CONTEXT_KEYS,
    SOURCE_SWEEP_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (
    SCHEMA_VERSION as SOURCE_FOLLOWUP_SCHEMA_VERSION,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input import (
    SCHEMA_VERSION as SOURCE_INPUT_GUARD_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package import (
    SCHEMA_VERSION as SOURCE_PACKAGE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio import (
    SCHEMA_VERSION as SOURCE_AUDIO_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    SCHEMA_VERSION as CHORD_TONE_REPAIR_SWEEP_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (
    SCHEMA_VERSION as CHORD_CONTEXT_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
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
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
}
AUDIO_SOURCE_SCHEMA_CONTEXT = {
    "source_schema_version": SOURCE_SWEEP_SCHEMA_VERSION,
    "source_followup_schema_version": SOURCE_FOLLOWUP_SCHEMA_VERSION,
    "source_objective_input_guard_schema_version": SOURCE_INPUT_GUARD_SCHEMA_VERSION,
    "source_package_schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
    "source_audio_schema_version": SOURCE_AUDIO_SCHEMA_VERSION,
    "chord_tone_repair_sweep_schema_version": CHORD_TONE_REPAIR_SWEEP_SCHEMA_VERSION,
    "chord_tone_repair_sweep_source_schema_version": CHORD_CONTEXT_OBJECTIVE_SCHEMA_VERSION,
    "chord_tone_repair_sweep_bridge_schema_version": BRIDGE_SCHEMA_VERSION,
}


def write_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate([60, 63, 67, 70] * 6):
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
                "rank": index,
                "source_midi_path": str(midi_path),
                "repaired_midi_path": str(midi_path),
                "changed_note_count": 1 if index in {3, 5} else 0,
                "after_bridge_flags": [],
                "after_chord_tone_ratio": 0.5,
                "after_cadence_landing_chord_tone": True,
                "after_max_non_chord_tone_run": 3,
                "wav_file": wav_file,
            }
        )
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_boundary": SOURCE_BOUNDARY,
        **AUDIO_SOURCE_SCHEMA_CONTEXT,
        "audio_render_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 6,
            "technical_wav_validation": True,
            "outside_soloing_repair_audio_package_completed": True,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
            **AUDIO_SOURCE_SCHEMA_CONTEXT,
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
            "changed_note_total": 2,
            "source_objective_outside_soloing_pitch_role_risk_count": 5,
            "source_outside_soloing_pitch_role_risk_count_before": 5,
            "source_outside_soloing_pitch_role_risk_count_after": 2,
            "source_outside_soloing_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_targeted": False,
            "source_outside_soloing_residual_risk_preserved": True,
            "outside_soloing_pitch_role_risk_count_before": 2,
            "outside_soloing_pitch_role_risk_count_after": 0,
            "outside_soloing_pitch_role_risk_delta": 2,
            "outside_soloing_repair_targeted": True,
            "weak_chord_tone_landing_risk_count_after": 0,
            "final_landing_chord_tone_count_after": 6,
            "max_non_chord_tone_run_before": 4,
            "max_non_chord_tone_run_after": 3,
            "target_supported": True,
            "audio_review_required": True,
            **AUDIO_SOURCE_SCHEMA_CONTEXT,
            **SOURCE_CONTEXT,
        },
        "rendered_audio_files": rendered,
    }


class StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageTest(
    unittest.TestCase
):
    def test_builds_pending_listening_review_package_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_package_report(
                audio_package_report=audio_package_report(root),
                output_dir=root / "review_package",
                issue_number=1144,
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
            self.assertEqual(
                summary["source_objective_outside_soloing_pitch_role_risk_count"], 5
            )
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_before"], 5)
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_after"], 2)
            self.assertEqual(summary["source_outside_soloing_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_targeted"])
            self.assertTrue(summary["source_outside_soloing_residual_risk_preserved"])
            self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 0)
            self.assertTrue(summary["outside_soloing_repair_targeted"])
            self.assertEqual(summary["weak_chord_tone_landing_risk_count_after"], 0)
            self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
            self.assertEqual(summary["max_non_chord_tone_run_after"], 3)
            self.assertEqual(report["schema_version"], REVIEW_SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1144)
            self.assertEqual(report["source_schema_version"], SOURCE_SCHEMA_VERSION)
            self.assertEqual(summary["source_schema_version"], SOURCE_SCHEMA_VERSION)
            self.assertEqual(
                summary["source_repair_sweep_schema_version"], SOURCE_SWEEP_SCHEMA_VERSION
            )
            self.assertEqual(
                summary["source_followup_schema_version"], SOURCE_FOLLOWUP_SCHEMA_VERSION
            )
            for key in SOURCE_PACKAGE_SCHEMA_CONTEXT_KEYS:
                self.assertEqual(report["source_summary"][key], summary[key])
                self.assertEqual(report["readiness"][key], summary[key])
            for key, value in SOURCE_CONTEXT.items():
                self.assertEqual(report["source_summary"][key], value)
                self.assertEqual(report["readiness"][key], value)
                self.assertEqual(summary[key], value)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_audio_package_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=audio_package_report(root, quality_claim=True),
                    output_dir=root / "review_package",
                    issue_number=1144,
                    expected_count=6,
                )

    def test_rejects_audio_package_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = audio_package_report(root)
            source["schema_version"] = "stale_schema"
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=source,
                    output_dir=root / "review_package",
                    issue_number=1144,
                    expected_count=6,
                )

    def test_rejects_audio_package_schema_context_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = audio_package_report(root)
            source["audio_render_boundary"][
                SOURCE_AUDIO_SCHEMA_CONTEXT_KEYS[1]
            ] = "stale_schema"
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=source,
                    output_dir=root / "review_package",
                    issue_number=1144,
                    expected_count=6,
                )

    def test_rejects_missing_source_context_preserved_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = audio_package_report(root)
            source["summary"][
                "repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError
            ):
                build_listening_review_package_report(
                    audio_package_report=source,
                    output_dir=root / "review_package",
                    issue_number=1144,
                    expected_count=6,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_input_guard",
        )
        self.assertEqual(
            REVIEW_SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package_v4",
        )


if __name__ == "__main__":
    unittest.main()
