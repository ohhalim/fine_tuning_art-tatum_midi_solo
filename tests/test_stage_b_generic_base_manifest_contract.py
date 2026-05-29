from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_stage_b_generic_base_manifest_contract import (
    StageBGenericBaseManifestContractError,
    build_contract_report,
    validate_contract_report,
)
from scripts.build_jazz_training_manifests import build_manifest_payload


def row(path: str, *, artist: str, album: str, sha1: str, brad: bool = False) -> dict:
    return {
        "path": path,
        "sha1": sha1,
        "source": "studio",
        "artist": artist,
        "album": album,
        "is_brad_mehldau": brad,
        "duration_sec": 120.0,
        "non_drum_note_count": 100,
        "piano_program_note_ratio": 1.0,
        "max_note_duration_ratio": 0.02,
        "recommendation": "candidate",
    }


def readiness(*, phase4_ready: bool = True, broad_claim: bool = False, brad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_generic_base_readiness_audit_v1",
        "readiness": {
            "boundary": "stage_b_generic_base_readiness_audit",
            "phase4_prep_ready": phase4_ready,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": brad_claim,
            "production_ready_improviser_claimed": False,
        },
    }


def manifest_payload() -> dict:
    rows = [
        row("a1.mid", artist="A", album="One", sha1="a1"),
        row("a2.mid", artist="A", album="Two", sha1="a2"),
        row("b1.mid", artist="B", album="One", sha1="b1"),
        row("b2.mid", artist="B", album="Two", sha1="b2"),
        row("brad1.mid", artist="Brad Mehldau", album="Live", sha1="c1", brad=True),
        row("brad2.mid", artist="Brad Mehldau", album="Studio", sha1="c2", brad=True),
        row("brad3.mid", artist="Brad Mehldau", album="Trio", sha1="c3", brad=True),
    ]
    return build_manifest_payload(
        {
            "summary": {
                "candidate_non_brad_file_count": 4,
                "candidate_brad_file_count": 3,
                "duplicate_exact_hash_group_count": 0,
                "duplicate_exact_file_count": 0,
            },
            "files": rows,
        },
        audit_json=Path("audit.json"),
        seed=1,
        generic_train_ratio=0.75,
        generic_val_ratio=0.25,
        brad_train_ratio=0.34,
        brad_val_ratio=0.33,
        brad_holdout_ratio=0.33,
        group_fields=["artist", "album"],
    )


class StageBGenericBaseManifestContractTest(unittest.TestCase):
    def test_manifest_contract_is_ready_without_claiming_training_quality(self) -> None:
        report = build_contract_report(
            manifest_payload(),
            readiness(),
            output_dir=Path("outputs/manifest_contract"),
            min_generic_train=1,
            min_generic_val=1,
            min_brad_holdout=1,
        )
        summary = validate_contract_report(
            report,
            expected_boundary="stage_b_generic_base_manifest_contract",
            expected_next_boundary="stage_b_generic_stage_b_window_prepare_smoke",
            require_contract_ready=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["manifest_contract_ready"])
        self.assertTrue(summary["stage_b_window_prepare_smoke_ready"])
        self.assertFalse(summary["broad_training_execution_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertFalse(summary["brad_style_adaptation_claimed"])
        self.assertEqual(summary["generic_brad_leak_count"], 0)
        self.assertEqual(summary["brad_non_brad_leak_count"], 0)
        self.assertEqual(summary["overlap_path_count"], 0)

    def test_rejects_readiness_claims(self) -> None:
        with self.assertRaises(StageBGenericBaseManifestContractError):
            build_contract_report(
                manifest_payload(),
                readiness(broad_claim=True),
                output_dir=Path("outputs/manifest_contract"),
                min_generic_train=1,
                min_generic_val=1,
                min_brad_holdout=1,
            )

    def test_requires_phase4_readiness(self) -> None:
        with self.assertRaises(StageBGenericBaseManifestContractError):
            build_contract_report(
                manifest_payload(),
                readiness(phase4_ready=False),
                output_dir=Path("outputs/manifest_contract"),
                min_generic_train=1,
                min_generic_val=1,
                min_brad_holdout=1,
            )

    def test_contract_not_ready_when_holdout_is_too_small(self) -> None:
        report = build_contract_report(
            manifest_payload(),
            readiness(),
            output_dir=Path("outputs/manifest_contract"),
            min_generic_train=1,
            min_generic_val=1,
            min_brad_holdout=5,
        )

        self.assertFalse(report["readiness"]["manifest_contract_ready"])


if __name__ == "__main__":
    unittest.main()
