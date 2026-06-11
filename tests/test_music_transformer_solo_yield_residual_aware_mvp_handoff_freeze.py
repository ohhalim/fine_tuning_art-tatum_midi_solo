from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_music_transformer_solo_yield_residual_aware_mvp_handoff import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareMvpHandoffFreezeError,
    build_handoff_freeze_report,
    validate_handoff_freeze_report,
)


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_text(path)


def final_review_package(root: Path, *, quality_claim: bool = False, bad_checksum: bool = False) -> dict:
    output_dir = root / "final_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "residual_aware_final_review_package.md").write_text("review", encoding="utf-8")
    (output_dir / "residual_aware_final_review_package.json").write_text("{}", encoding="utf-8")
    (output_dir / "residual_aware_review_input_template.json").write_text("{}", encoding="utf-8")
    midi_sha = write_file(root / "candidate_01.mid", "midi")
    wav_sha = write_file(root / "candidate_01.wav", "wav")
    if bad_checksum:
        midi_sha = "0" * 64
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_final_review_package_v1",
        "output_dir": str(output_dir),
        "aggregate": {
            "candidate_count": 1,
            "midi_count": 1,
            "wav_count": 1,
            "quality_proxy_pass_count": 1,
            "quality_proxy_fail_count": 0,
            "major_label_counts": {},
            "watch_label_counts": {},
            "missing_file_count": 0,
            "checksum_mismatch_count": 0,
        },
        "candidate_handoff": [
            {
                "review_index": 1,
                "case_label": "unit",
                "sample_index": 1,
                "sample_seed": 10,
                "review_midi_path": str(root / "candidate_01.mid"),
                "review_wav_path": str(root / "candidate_01.wav"),
                "review_midi_sha256": midi_sha,
                "review_wav_sha256": wav_sha,
                "duration_seconds": 1.0,
                "sample_rate": 44100,
                "quality_proxy_pass": True,
                "rubric_major_labels": [],
                "rubric_watch_labels": [],
            }
        ],
        "readiness": {
            "residual_aware_final_review_package_ready": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def input_guard() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_listening_input_guard_v1",
        "output_dir": "outputs/input_guard",
        "guard_result": {
            "review_item_count": 1,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
        },
        "readiness": {
            "residual_aware_listening_input_guard_completed": True,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def status_audit(*, synced: bool = True) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_status_audit_v1",
        "output_dir": "outputs/status_audit",
        "aggregate": {
            "candidate_count": 1,
            "midi_count": 1,
            "wav_count": 1,
            "quality_proxy_pass_count": 1,
            "quality_proxy_fail_count": 0,
        },
        "readiness": {
            "residual_aware_status_audit_completed": True,
            "residual_aware_status_synced": synced,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_mvp_handoff_freeze",
            "critical_user_input_required": False,
        },
    }


class MusicTransformerSoloYieldResidualAwareMvpHandoffFreezeTest(unittest.TestCase):
    def test_builds_handoff_freeze_with_verified_local_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            report = build_handoff_freeze_report(
                final_review_package=final_review_package(root),
                input_guard_report=input_guard(),
                status_audit_report=status_audit(),
                output_dir=root / "handoff",
                issue_number=1396,
            )
        summary = validate_handoff_freeze_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_local_artifacts_verified=True,
            require_pending_input=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["local_candidate_artifacts_verified"])
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["midi_count"], 1)
        self.assertEqual(summary["wav_count"], 1)
        self.assertEqual(summary["checksum_mismatch_count"], 0)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["raw_artifact_upload_required"])

    def test_rejects_unsynced_status_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualAwareMvpHandoffFreezeError):
                build_handoff_freeze_report(
                    final_review_package=final_review_package(root),
                    input_guard_report=input_guard(),
                    status_audit_report=status_audit(synced=False),
                    output_dir=root / "handoff",
                    issue_number=1396,
                )

    def test_rejects_candidate_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualAwareMvpHandoffFreezeError):
                build_handoff_freeze_report(
                    final_review_package=final_review_package(root, bad_checksum=True),
                    input_guard_report=input_guard(),
                    status_audit_report=status_audit(),
                    output_dir=root / "handoff",
                    issue_number=1396,
                )

    def test_rejects_quality_claim_in_final_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            with self.assertRaises(SoloYieldResidualAwareMvpHandoffFreezeError):
                build_handoff_freeze_report(
                    final_review_package=final_review_package(root, quality_claim=True),
                    input_guard_report=input_guard(),
                    status_audit_report=status_audit(),
                    output_dir=root / "handoff",
                    issue_number=1396,
                )


if __name__ == "__main__":
    unittest.main()
