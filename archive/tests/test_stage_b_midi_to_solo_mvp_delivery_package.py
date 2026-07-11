from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_mvp_delivery_package import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloMvpDeliveryPackageError,
    build_delivery_package_report,
    validate_delivery_package_report,
)
from scripts.decide_stage_b_midi_to_solo_listening_review_quality_gap import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY as LISTENING_GAP_BOUNDARY,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    NEXT_BOUNDARY as LISTENING_GAP_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as LISTENING_GAP_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (
    BOUNDARY as CLI_PACKAGE_BOUNDARY,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return str(path)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
}


def listening_gap_report(
    *,
    quality_claim: bool = False,
    outside_soloing_repair_supported: bool = True,
) -> dict:
    return {
        "schema_version": LISTENING_GAP_SCHEMA_VERSION,
        "boundary": LISTENING_GAP_BOUNDARY,
        "source_schema_versions": {
            "quality_gap_decision": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "current_evidence": CURRENT_EVIDENCE_SCHEMA_VERSION,
        },
        "source_schema_version": QUALITY_GAP_DECISION_SCHEMA_VERSION,
        "source_current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
        "quality_gap_summary": {
            "current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
            "technical_model_core_mvp_completed": True,
            "changed_ratio_repair_objective_completed": True,
            "outside_soloing_repair_objective_completed": outside_soloing_repair_supported,
            "outside_soloing_repair_source_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "rendered_audio_file_count": 3,
            "max_repaired_interval": 12,
            "max_interval_threshold": 12,
            "max_repaired_pitch_changed_ratio": 0.4348,
            "target_max_pitch_changed_ratio": 0.5,
            "listening_review_quality_gap_open": True,
            "outside_soloing_repair_objective_path_ready": outside_soloing_repair_supported,
            "outside_soloing_repair_target_supported": outside_soloing_repair_supported,
            "outside_soloing_repair_rendered_audio_file_count": 6,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "outside_soloing_repair_source_targeted": False,
            "outside_soloing_repair_source_residual_risk_preserved": True,
            "outside_soloing_repair_pitch_role_risk_count_after": (
                0 if outside_soloing_repair_supported else 1
            ),
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            **SOURCE_CONTEXT,
        },
        "readiness": {
            "listening_review_quality_gap_completed": True,
            "technical_mvp_delivery_package_ready": True,
            "outside_soloing_repair_source_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
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
                issue_number=1160,
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
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["source_listening_gap_schema_version"], LISTENING_GAP_SCHEMA_VERSION)
        self.assertEqual(
            summary["source_quality_gap_schema_version"],
            QUALITY_GAP_DECISION_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["changed_ratio_repair_wav_count"], 3)
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(
            summary["outside_soloing_repair_source_objective_pitch_role_risk_count"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_before"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_after"],
            2,
        )
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["outside_soloing_repair_source_targeted"])
        self.assertTrue(summary["outside_soloing_repair_source_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertTrue(summary["listening_review_quality_gap_open"])
        self.assertFalse(summary["raw_artifact_upload_required"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo README final evidence refresh source-context refresh",
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            report["source_schema_versions"]["listening_review_quality_gap"],
            LISTENING_GAP_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["source_schema_versions"]["quality_gap_decision"],
            QUALITY_GAP_DECISION_SCHEMA_VERSION,
        )
        self.assertEqual(report["issue_number"], 1160)

    def test_rejects_listening_gap_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["schema_version"] = "stale_schema"
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=1160,
                )

    def test_rejects_listening_gap_source_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["source_schema_version"] = "stale_schema"
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=1160,
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

    def test_rejects_missing_outside_soloing_repair_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=listening_gap_report(
                        outside_soloing_repair_supported=False
                    ),
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=822,
                )

    def test_rejects_missing_outside_soloing_source_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"].pop(
                "outside_soloing_repair_source_residual_risk_preserved"
            )
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=908,
                )

    def test_rejects_missing_outside_soloing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=992,
                )

    def test_rejects_false_outside_soloing_source_context_preserved_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"][
                "repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=1076,
                )

    def test_rejects_false_outside_soloing_schema_context_preserved_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"]["outside_soloing_repair_schema_context_preserved"] = False
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=1160,
                )

    def test_rejects_delivery_package_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = build_delivery_package_report(
                listening_review_quality_gap=listening_gap_report(),
                cli_mvp_package=cli_package_report(tmp_path),
                changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                output_dir=tmp_path / "delivery",
                issue_number=1160,
            )
            report["schema_version"] = "stale_schema"
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                validate_delivery_package_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    require_delivery_completed=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_targeted_outside_soloing_source_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"]["outside_soloing_repair_source_targeted"] = True
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=908,
                )

    def test_rejects_outside_soloing_source_risk_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = listening_gap_report()
            source["quality_gap_summary"][
                "outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 2
            with self.assertRaises(StageBMidiToSoloMvpDeliveryPackageError):
                build_delivery_package_report(
                    listening_review_quality_gap=source,
                    cli_mvp_package=cli_package_report(tmp_path),
                    changed_ratio_audio_package=changed_ratio_audio_report(tmp_path),
                    output_dir=tmp_path / "delivery",
                    issue_number=908,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_delivery_package")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_readme_final_evidence_refresh")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_mvp_delivery_package_v4")


if __name__ == "__main__":
    unittest.main()
