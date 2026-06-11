from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from scripts.audit_music_transformer_solo_yield_interval_contour_handoff_reproducibility import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldIntervalContourHandoffReproducibilityAuditError,
    build_reproducibility_audit,
    validate_report,
)
from scripts.build_music_transformer_solo_yield_interval_contour_final_review_handoff import (
    SCHEMA_VERSION as HANDOFF_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file, wav_meta


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(b"\x00\x00\x00\x00" * 441)


def write_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0")


def handoff_report(root: Path, *, bad_midi_checksum: bool = False, residual_count: int = 0) -> dict:
    midi_path = root / "candidate.mid"
    wav_path = root / "candidate.wav"
    write_midi(midi_path)
    write_wav(wav_path)
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "output_dir": str(root),
        "boundary": "music_transformer_solo_yield_interval_contour_final_review_handoff",
        "candidate_handoff": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": "bad" if bad_midi_checksum else sha256_file(midi_path),
                "review_wav_sha256": str(wav_meta(wav_path)["sha256"]),
                "duration_seconds": float(wav_meta(wav_path)["duration_seconds"]),
                "sample_rate": 44100,
                "objective_profile": {
                    "midi_note_count": 33,
                    "midi_chord_tone_ratio": 0.52,
                    "midi_max_gap_seconds": 0.60,
                    "midi_direction_change_ratio": 0.58,
                    "midi_max_abs_interval": 7,
                    "final_landing_chord": "Ebmaj7",
                    "final_landing_is_chord_tone": True,
                },
                "residual_labels": ["wide_interval_review"] if residual_count else [],
            }
        ],
        "aggregate": {
            "candidate_count": 1,
            "midi_count": 1,
            "wav_count": 1,
            "technical_wav_validation": True,
            "objective_residual_label_count": residual_count,
        },
        "readiness": {
            "final_review_handoff_ready": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "selected_next_target": "manual_listening_review_pending",
            "next_boundary": "music_transformer_solo_yield_interval_contour_aftercare_listening_review",
        },
    }


class MusicTransformerSoloYieldIntervalContourHandoffReproducibilityAuditTest(unittest.TestCase):
    def test_builds_reproducibility_audit_for_matching_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_reproducibility_audit(
                handoff_report(root),
                output_dir=root / "out",
                expected_candidate_count=1,
                expected_residual_label_count=0,
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["missing_midi_count"], 0)
        self.assertEqual(summary["missing_wav_count"], 0)
        self.assertEqual(summary["midi_checksum_mismatch_count"], 0)
        self.assertEqual(summary["wav_checksum_mismatch_count"], 0)
        self.assertEqual(summary["objective_residual_label_count"], 0)
        self.assertTrue(summary["reproducible_handoff"])
        self.assertEqual(summary["selected_next_target"], "sampling_repeatability_audit")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldIntervalContourHandoffReproducibilityAuditError):
                build_reproducibility_audit(
                    handoff_report(root, bad_midi_checksum=True),
                    output_dir=root / "out",
                    expected_candidate_count=1,
                    expected_residual_label_count=0,
                )

    def test_rejects_residual_label_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldIntervalContourHandoffReproducibilityAuditError):
                build_reproducibility_audit(
                    handoff_report(root, residual_count=1),
                    output_dir=root / "out",
                    expected_candidate_count=1,
                    expected_residual_label_count=0,
                )


if __name__ == "__main__":
    unittest.main()
