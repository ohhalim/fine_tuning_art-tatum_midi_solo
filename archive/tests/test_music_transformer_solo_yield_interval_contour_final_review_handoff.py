from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_music_transformer_solo_yield_interval_contour_aftercare_listening_package import (
    INPUT_SCHEMA_VERSION,
    SCHEMA_VERSION as LISTENING_PACKAGE_SCHEMA_VERSION,
)
from scripts.build_music_transformer_solo_yield_interval_contour_final_review_handoff import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SoloYieldIntervalContourFinalReviewHandoffError,
    build_handoff_package,
    validate_report,
)
from scripts.decide_music_transformer_solo_yield_interval_contour_aftercare_objective_next import (
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.render_music_transformer_solo_yield_interval_contour_aftercare_audio import (
    SCHEMA_VERSION as AUDIO_PACKAGE_SCHEMA_VERSION,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def wav_meta(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": 4,
        "sha256": "abc123",
        "channels": 2,
        "sample_width_bytes": 2,
        "sample_rate": 44100,
        "frame_count": 44100,
        "duration_seconds": 1.0,
    }


def listening_package(root: Path, *, quality_claim: bool = False) -> dict:
    midi_path = root / "review" / "midi" / "candidate_01.mid"
    wav_path = root / "review" / "audio" / "candidate_01.wav"
    touch(midi_path)
    touch(wav_path)
    return {
        "schema_version": LISTENING_PACKAGE_SCHEMA_VERSION,
        "output_dir": str(root / "review"),
        "candidate_count": 1,
        "candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "review_midi_path": str(midi_path),
                "review_wav_path": str(wav_path),
                "review_midi_sha256": "midi123",
                "review_wav_file": wav_meta(wav_path),
            }
        ],
        "review_input_template": {
            "schema_version": INPUT_SCHEMA_VERSION,
            "review_status": "pending",
            "overall_decision": "pending",
            "candidates": [{"review_index": 1, "decision": "pending"}],
        },
        "readiness": {
            "listening_package_ready": True,
            "candidate_midi_files_copied": 1,
            "candidate_wav_files_copied": 1,
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def audio_package(root: Path) -> dict:
    midi_path = root / "source" / "candidate_01.mid"
    wav_path = root / "source" / "candidate_01.wav"
    touch(midi_path)
    touch(wav_path)
    return {
        "schema_version": AUDIO_PACKAGE_SCHEMA_VERSION,
        "output_dir": str(root / "source"),
        "rendered_audio_files": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "repaired_midi_path": str(midi_path),
                "wav_file": wav_meta(wav_path),
            }
        ],
        "aggregate": {
            "rendered_wav_count": 1,
            "technical_wav_validation": True,
        },
        "readiness": {
            "audio_package_completed": True,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def objective_decision() -> dict:
    return {
        "schema_version": OBJECTIVE_DECISION_SCHEMA_VERSION,
        "output_dir": "outputs/objective",
        "candidate_residuals": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "after_profile": {
                    "midi_note_count": 33,
                    "midi_chord_tone_ratio": 0.52,
                    "midi_max_gap_seconds": 0.60,
                    "midi_direction_change_ratio": 0.58,
                    "midi_max_abs_interval": 7,
                    "final_landing_chord": "Ebmaj7",
                    "final_landing_is_chord_tone": True,
                },
                "residual_labels": [],
            }
        ],
        "aggregate": {
            "candidate_count": 1,
            "final_landing_not_chord_tone_count": 0,
            "midi_low_chord_tone_ratio_count": 0,
            "dead_air_aftercare_count": 0,
            "weak_direction_change_count": 0,
            "low_note_count_for_4bar_count": 0,
            "wide_interval_review_count": 0,
            "midi_chord_tone_ratio_min": 0.52,
            "midi_chord_tone_ratio_avg": 0.52,
            "midi_note_count_min": 33,
            "midi_note_count_max": 33,
            "midi_max_gap_seconds_max": 0.60,
            "midi_max_abs_interval_max": 7,
        },
        "decision": {
            "selected_next_target": "listening_review_required",
            "next_boundary": "music_transformer_solo_yield_interval_contour_aftercare_listening_review",
        },
        "readiness": {
            "objective_only_next_decision_completed": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldIntervalContourFinalReviewHandoffTest(unittest.TestCase):
    def test_builds_final_handoff_for_ready_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_handoff_package(
                listening_package(root),
                audio_package(root),
                objective_decision(),
                output_dir=root / "out",
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["midi_count"], 1)
        self.assertEqual(summary["wav_count"], 1)
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["objective_residual_label_count"], 0)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["selected_next_target"], "manual_listening_review_pending")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim_from_source_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldIntervalContourFinalReviewHandoffError):
                build_handoff_package(
                    listening_package(root, quality_claim=True),
                    audio_package(root),
                    objective_decision(),
                    output_dir=root / "out",
                )

    def test_rejects_objective_residual_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            objective = objective_decision()
            objective["candidate_residuals"][0]["residual_labels"] = ["wide_interval_review"]
            with self.assertRaises(SoloYieldIntervalContourFinalReviewHandoffError):
                build_handoff_package(
                    listening_package(root),
                    audio_package(root),
                    objective,
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
