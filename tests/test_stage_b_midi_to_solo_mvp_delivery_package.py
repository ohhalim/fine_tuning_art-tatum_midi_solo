from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloMvpDeliveryPackageError,
    build_delivery_package_report,
    validate_delivery_package_report,
)
from scripts.decide_stage_b_midi_to_solo_listening_review_quality_gap import (
    BOUNDARY as LISTENING_GAP_BOUNDARY,
    NEXT_BOUNDARY as LISTENING_GAP_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (
    BOUNDARY as CLI_PACKAGE_BOUNDARY,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return str(path)


def listening_gap_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": LISTENING_GAP_BOUNDARY,
        "quality_gap_summary": {
            "technical_model_core_mvp_completed": True,
            "changed_ratio_repair_objective_completed": True,
            "rendered_audio_file_count": 3,
            "max_repaired_interval": 12,
            "max_interval_threshold": 12,
            "max_repaired_pitch_changed_ratio": 0.4348,
            "target_max_pitch_changed_ratio": 0.5,
            "listening_review_quality_gap_open": True,
        },
        "readiness": {
            "listening_review_quality_gap_completed": True,
            "technical_mvp_delivery_package_ready": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": LISTENING_GAP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def cli_package_report(tmp_path: Path, *, quality_claim: bool = False) -> dict:
    input_midi = touch(tmp_path / "input" / "fixture.mid")
    candidates = []
    for rank in range(1, 4):
        candidates.append(
            {
                "rank": rank,
                "repaired_midi_path": touch(tmp_path / "cli" / f"rank_{rank:02d}.mid"),
                "note_count": 96,
                "unique_pitch_count": 20 + rank,
                "dead_air_ratio": 0.18 + (rank * 0.01),
                "objective_supported": True,
            }
        )
    return {
        "boundary": CLI_PACKAGE_BOUNDARY,
        "input": {"midi_path": input_midi},
        "cli": {
            "script": "scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py",
            "command": ".venv/bin/python scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py --input_midi input.mid",
        },
        "objective_summary": {
            "candidate_count": 3,
            "objective_supported_candidate_count": 3,
            "cli_mvp_package_ready": True,
            "min_dead_air_ratio": 0.19,
            "max_dead_air_ratio": 0.22,
        },
        "candidate_manifest": candidates,
        "readiness": {
            "cli_mvp_package_completed": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {"critical_user_input_required": False},
    }


def changed_ratio_audio_report(tmp_path: Path, *, rendered_count: int = 3) -> dict:
    rendered = []
    for rank in range(1, rendered_count + 1):
        rendered.append(
            {
                "rank": rank,
                "repaired_midi_path": touch(tmp_path / "changed_ratio" / f"rank_{rank:02d}.mid"),
                "repaired_max_interval": 12,
                "repaired_unique_pitch_count": 23 + rank,
                "pitch_changed_ratio": 0.43 - (rank * 0.02),
                "wav_file": {
                    "path": touch(tmp_path / "changed_ratio" / f"rank_{rank:02d}.wav"),
                    "duration_seconds": 18.4 + rank,
                    "sample_rate": 44100,
                },
            }
        )
    return {
        "summary": {
            "rendered_audio_file_count": rendered_count,
            "technical_wav_validation": True,
            "duration_min_seconds": 18.4,
            "duration_max_seconds": 19.0,
        },
        "source_summary": {
            "changed_ratio_repair_passed": True,
            "repaired_max_pitch_changed_ratio": 0.4348,
            "max_pitch_changed_ratio": 0.5,
            "repaired_max_interval": 12,
            "target_max_interval": 12,
        },
        "rendered_audio_files": rendered,
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package",
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloMvpDeliveryPackageTest(unittest.TestCase):
    def test_builds_delivery_package_manifest_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = build_delivery_package_report(
                listening_review_quality_gap=listening_gap_report(),
                cli_mvp_package=cli_package_report(tmp_path),
                changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                output_dir=tmp_path / "delivery",
                issue_number=738,
            )
            summary = validate_delivery_package_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_delivery_completed=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["mvp_delivery_package_completed"])
        self.assertTrue(summary["runnable_cli_ready"])
        self.assertTrue(summary["input_to_ranked_midi_ready"])
        self.assertTrue(summary["input_to_rendered_wav_evidence_ready"])
        self.assertTrue(summary["changed_ratio_repair_audio_evidence_ready"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["changed_ratio_repair_wav_count"], 3)
        self.assertTrue(summary["listening_review_quality_gap_open"])
        self.assertFalse(summary["raw_artifact_upload_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo README final evidence refresh",
        )

    def test_rejects_listening_gap_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=listening_gap_report(quality_claim=True),
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=738,
                )

    def test_rejects_cli_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=listening_gap_report(),
                    cli_mvp_package=cli_package_report(tmp_path, quality_claim=True),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=738,
                )

    def test_rejects_missing_changed_ratio_audio_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=listening_gap_report(),
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path, rendered_count=2),
                    output_dir=tmp_path / "delivery",
                    issue_number=738,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_delivery_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_readme_final_evidence_refresh")


if __name__ == "__main__":
    unittest.main()
