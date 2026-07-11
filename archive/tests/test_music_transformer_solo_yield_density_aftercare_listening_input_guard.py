from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.assess_stage_b_generic_base_readiness import write_json
from scripts.guard_music_transformer_solo_yield_density_aftercare_listening_input import (
    BOUNDARY,
    INPUT_SCHEMA_VERSION,
    NEXT_BOUNDARY_PENDING,
    NEXT_BOUNDARY_REVIEWED,
    SCHEMA_VERSION,
    SOURCE_PACKAGE_SCHEMA_VERSION,
    SoloYieldDensityAftercareListeningInputGuardError,
    build_guard_report,
    validate_report,
)


def input_template(*, reviewed: bool = False) -> dict:
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "review_status": "reviewed" if reviewed else "pending",
        "overall_decision": "keep_candidate_1" if reviewed else "pending",
        "preferred_review_index": 1 if reviewed else None,
        "candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "decision": "keep" if reviewed else "pending",
                "usable_as_jazz_solo_phrase": True if reviewed else None,
                "primary_failure": "none" if reviewed else None,
                "notes": "",
            },
            {
                "review_index": 2,
                "case_label": "dominant_cycle",
                "decision": "reject" if reviewed else "pending",
                "usable_as_jazz_solo_phrase": False if reviewed else None,
                "primary_failure": "too_mechanical" if reviewed else None,
                "notes": "",
            },
        ],
    }


def package_report(root: Path, *, quality_claim: bool = False, reviewed: bool = False) -> dict:
    package_dir = root / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    write_json(package_dir / "listening_review_input_template.json", input_template(reviewed=reviewed))
    return {
        "schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "output_dir": str(package_dir),
        "candidate_count": 2,
        "readiness": {
            "listening_package_ready": True,
            "candidate_midi_files_copied": 2,
            "candidate_wav_files_copied": 2,
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldDensityAftercareListeningInputGuardTest(unittest.TestCase):
    def test_pending_template_blocks_preference_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_guard_report(
                package_report(root),
                output_dir=root / "guard",
                listening_input_path=None,
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["boundary"], BOUNDARY)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertTrue(summary["schema_matched"])
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertTrue(summary["objective_only_next_decision_required"])
        self.assertEqual(summary["pending_candidate_field_count"], 6)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY_PENDING)

    def test_reviewed_template_allows_preference_fill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_guard_report(
                package_report(root, reviewed=True),
                output_dir=root / "guard",
                listening_input_path=None,
            )
        summary = validate_report(report, require_no_quality_claim=True)

        self.assertTrue(summary["validated_listening_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])
        self.assertFalse(summary["objective_only_next_decision_required"])
        self.assertEqual(summary["pending_candidate_field_count"], 0)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY_REVIEWED)

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldDensityAftercareListeningInputGuardError):
                build_guard_report(
                    package_report(root, quality_claim=True),
                    output_dir=root / "guard",
                    listening_input_path=None,
                )


if __name__ == "__main__":
    unittest.main()
