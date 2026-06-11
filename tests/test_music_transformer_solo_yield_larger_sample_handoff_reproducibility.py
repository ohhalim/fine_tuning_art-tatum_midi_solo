from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from scripts.audit_music_transformer_solo_yield_larger_sample_handoff_reproducibility import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldLargerSampleHandoffReproducibilityAuditError,
    build_reproducibility_audit,
    validate_report,
)
from scripts.build_music_transformer_solo_yield_larger_sample_final_review_handoff import (
    SCHEMA_VERSION as HANDOFF_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file


def touch(path: Path, payload: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 4410)


def handoff_report(root: Path, *, quality_claim: bool = False, bad_checksum: bool = False) -> dict:
    midi_path = root / "handoff" / "midi" / "candidate_01.mid"
    wav_path = root / "handoff" / "audio" / "candidate_01.wav"
    touch(midi_path, b"midi")
    write_wav(wav_path)
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "output_dir": str(root / "handoff"),
        "boundary": "music_transformer_solo_yield_larger_sample_final_review_handoff",
        "candidate_handoff": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": "bad" if bad_checksum else sha256_file(midi_path),
                "review_wav_sha256": sha256_file(wav_path),
                "selected_by_objective": True,
            }
        ],
        "aggregate": {
            "candidate_count": 1,
            "midi_count": 1,
            "wav_count": 1,
            "selected_objective_candidate_count": 1,
            "source_strict_valid_sample_count": 1,
            "source_sample_count": 1,
            "source_strict_yield_rate": 1.0,
            "missing_file_count": 0,
            "checksum_mismatch_count": 0,
        },
        "readiness": {
            "final_review_handoff_ready": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "selected_next_target": "manual_listening_review_pending",
            "next_boundary": "music_transformer_solo_yield_larger_sample_listening_review",
        },
    }


class MusicTransformerSoloYieldLargerSampleHandoffReproducibilityTest(unittest.TestCase):
    def test_builds_reproducibility_audit_for_ready_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_reproducibility_audit(
                handoff_report(root),
                output_dir=root / "out",
                expected_candidate_count=1,
                expected_selected_count=1,
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["selected_objective_candidate_count"], 1)
        self.assertEqual(summary["missing_midi_count"], 0)
        self.assertEqual(summary["missing_wav_count"], 0)
        self.assertEqual(summary["midi_checksum_mismatch_count"], 0)
        self.assertEqual(summary["wav_checksum_mismatch_count"], 0)
        self.assertTrue(summary["reproducible_handoff"])
        self.assertEqual(summary["selected_next_target"], "broader_repaired_sampling_repeatability_audit")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim_from_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldLargerSampleHandoffReproducibilityAuditError):
                build_reproducibility_audit(
                    handoff_report(root, quality_claim=True),
                    output_dir=root / "out",
                    expected_candidate_count=1,
                    expected_selected_count=1,
                )

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldLargerSampleHandoffReproducibilityAuditError):
                build_reproducibility_audit(
                    handoff_report(root, bad_checksum=True),
                    output_dir=root / "out",
                    expected_candidate_count=1,
                    expected_selected_count=1,
                )


if __name__ == "__main__":
    unittest.main()
