from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_music_transformer_solo_yield_larger_sample_final_review_handoff import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldLargerSampleFinalHandoffError,
    build_handoff_package,
    validate_report,
)
from scripts.build_music_transformer_solo_yield_listening_package import (
    SCHEMA_VERSION as LISTENING_PACKAGE_SCHEMA_VERSION,
)
from scripts.decide_music_transformer_solo_yield_objective_next import (
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file


def touch(path: Path, payload: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def wav_file(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "channels": 2,
        "sample_width_bytes": 2,
        "sample_rate": 44100,
        "frame_count": 44100,
        "duration_seconds": 1.0,
    }


def listening_package(root: Path, *, quality_claim: bool = False, bad_checksum: bool = False) -> dict:
    midi_path = root / "review" / "midi" / "candidate_01.mid"
    wav_path = root / "review" / "audio" / "candidate_01.wav"
    touch(midi_path, b"midi")
    touch(wav_path, b"wav")
    return {
        "schema_version": LISTENING_PACKAGE_SCHEMA_VERSION,
        "output_dir": str(root / "review"),
        "source_sweep": {
            "sample_count": 6,
            "strict_valid_sample_count": 6,
            "strict_yield_rate": 1.0,
        },
        "candidate_count": 1,
        "candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "chords": "Cm7,F7,Bbmaj7,Ebmaj7",
                "score": 233.0,
                "note_count": 31,
                "unique_pitch_count": 16,
                "dead_air_ratio": 0.62,
                "direction_change_ratio": 0.48,
                "syncopated_onset_ratio": 0.80,
                "chord_tone_ratio": 0.44,
                "tension_ratio": 0.25,
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": "bad" if bad_checksum else sha256_file(midi_path),
                "review_wav_file": wav_file(wav_path),
            }
        ],
        "review_input_template": {
            "schema_version": "music_transformer_solo_yield_listening_input_v1",
            "review_status": "pending",
            "overall_decision": "pending",
            "candidates": [{"review_index": 1, "decision": "pending"}],
        },
        "readiness": {
            "listening_review_package_ready": True,
            "candidate_midi_files_copied": 1,
            "candidate_wav_files_copied": 1,
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def objective_decision() -> dict:
    return {
        "schema_version": OBJECTIVE_DECISION_SCHEMA_VERSION,
        "output_dir": "outputs/objective",
        "objective_summary": {
            "candidate_count": 1,
            "score_min": 233.0,
            "score_max": 233.0,
            "score_avg": 233.0,
            "note_count_min": 31,
            "note_count_max": 31,
            "note_count_avg": 31.0,
            "dead_air_min": 0.62,
            "dead_air_max": 0.62,
            "dead_air_avg": 0.62,
        },
        "selected_objective_candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "score": 233.0,
                "note_count": 31,
                "dead_air_ratio": 0.62,
            }
        ],
        "readiness": {
            "objective_only_next_decision_completed": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_objective_only_next_decision",
            "next_boundary": BOUNDARY,
        },
    }


class MusicTransformerSoloYieldLargerSampleFinalReviewHandoffTest(unittest.TestCase):
    def test_builds_final_handoff_from_listening_package_and_objective_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_handoff_package(
                listening_package(root),
                objective_decision(),
                output_dir=root / "out",
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["midi_count"], 1)
        self.assertEqual(summary["wav_count"], 1)
        self.assertEqual(summary["selected_objective_candidate_count"], 1)
        self.assertEqual(summary["source_strict_valid_sample_count"], 6)
        self.assertEqual(summary["source_sample_count"], 6)
        self.assertEqual(summary["source_strict_yield_rate"], 1.0)
        self.assertEqual(summary["checksum_mismatch_count"], 0)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["selected_next_target"], "manual_listening_review_pending")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim_from_source_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldLargerSampleFinalHandoffError):
                build_handoff_package(
                    listening_package(root, quality_claim=True),
                    objective_decision(),
                    output_dir=root / "out",
                )

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldLargerSampleFinalHandoffError):
                build_handoff_package(
                    listening_package(root, bad_checksum=True),
                    objective_decision(),
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
